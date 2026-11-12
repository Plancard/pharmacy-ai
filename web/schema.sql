-- 피지컬 AI 약국 조제 보조 서비스 — DB 스키마 (PostgreSQL 16)
--
-- 규칙
--   ID     : 모듈·JSON·파일 경로에 나가는 ID(대화, 처방, 이미지, 메시지)는 UUID, 테이블 안에서만 쓰는 ID는 BIGINT IDENTITY
--   시각   : 모두 TIMESTAMPTZ (UTC로 저장, 화면에서 KST로 표시)
--   JSON   : 모듈 간 데이터(A.3~A.6)는 JSONB 원문 그대로, 필터에 쓰는 값만 컬럼으로 뽑는다
--   이력   : approvals, inspections, messages는 추가만 한다 (UPDATE 금지 트리거)
--   대화   : 대화 1개 = LangGraph thread 1개 (thread_id = conversations.id).
--            처방이 있는 대화는 prescriptions.id = conversations.id (같은 ID를 쓴다).
--            처방 없이 글로 시작한 상담 대화는 conversations 행만 있다.
--   체크포인트: LangGraph PostgresSaver.setup()이 같은 DB에 checkpoint* 테이블을 직접 만든다.
--             thread_id = conversations.id(UUID 문자열), FK 없이 값으로만 이어진다.
--
-- 앱 첫 실행 때 한 번 실행한다 (여러 번 실행해도 안전).

SET client_min_messages = warning;   -- 다시 실행할 때 "이미 있음" 안내를 숨긴다

-- ------------------------------------------------------------ 약국 · 사용자
CREATE TABLE IF NOT EXISTS pharmacies (
    id          BIGINT      GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    name        TEXT        NOT NULL,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS users (
    id            BIGINT      GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    pharmacy_id   BIGINT      NOT NULL REFERENCES pharmacies(id),
    username      TEXT        NOT NULL UNIQUE,
    password_hash TEXT        NOT NULL,                       -- werkzeug 해시
    role          TEXT        NOT NULL DEFAULT 'pharmacist'
                  CHECK (role IN ('pharmacist', 'admin')),    -- 1차는 pharmacist만 사용
    created_at    TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_users_pharmacy ON users(pharmacy_id);

-- ------------------------------------------------------------ 대화
-- 왼쪽 목록 1줄 = 대화 1개. 처방전 사진이 들어오면 같은 ID로 처방(prescriptions)이 생긴다.
-- 상담으로 시작한 대화에 처방전이 오면 그 대화가 처방 대화가 된다 (앞의 상담 기록이 이어진다).
CREATE TABLE IF NOT EXISTS conversations (
    id          UUID        PRIMARY KEY DEFAULT gen_random_uuid(),   -- LangGraph thread_id
    pharmacy_id BIGINT      NOT NULL REFERENCES pharmacies(id),
    user_id     BIGINT      NOT NULL REFERENCES users(id),           -- 대화를 시작한 사용자
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at  TIMESTAMPTZ NOT NULL DEFAULT now()                   -- 마지막 메시지 시각 (목록 정렬용)
);
CREATE INDEX IF NOT EXISTS idx_conversations_updated ON conversations(pharmacy_id, updated_at);

-- ------------------------------------------------------------ 처방
CREATE TABLE IF NOT EXISTS prescriptions (
    id          UUID        PRIMARY KEY DEFAULT gen_random_uuid(),   -- = conversations.id (아래 FK), A.3~A.6 prescription_id
    pharmacy_id BIGINT      NOT NULL REFERENCES pharmacies(id),
    display_no  TEXT        NOT NULL,                                -- 화면 표시용 번호 (예: P0001)
    user_id     BIGINT      NOT NULL REFERENCES users(id),
    status      TEXT        NOT NULL DEFAULT 'processing',          -- 값 목록은 아래 prescriptions_status_check
    hitl_point  TEXT        CHECK (hitl_point IN ('type_select', 'rx_approval', 'inspection', 'risk')),
    error_code  TEXT        CHECK (error_code IN ('E_OCR_EMPTY', 'E_TRAY_NOT_FOUND', 'E_CV_INFER', 'E_PLAN_RANGE',
                                                  'E_BARCODE_FAIL', 'E_LLM', 'E_EXT_API', 'E_DB')),
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (pharmacy_id, display_no),
    CHECK ((status = 'waiting_approval') = (hitl_point IS NOT NULL)),
    CHECK (status = 'failed' OR error_code IS NULL)
);
CREATE INDEX IF NOT EXISTS idx_prescriptions_status  ON prescriptions(pharmacy_id, status);
CREATE INDEX IF NOT EXISTS idx_prescriptions_created ON prescriptions(pharmacy_id, created_at);

-- 처방 상태 값 (기능분석서 5.2절). 값을 늘리면 이미 만든 DB에도 앱 시작(init-db) 때 반영되도록 따로 건다.
--   processing        그래프 실행 중
--   waiting_approval  HITL에서 약사 응답 대기 (hitl_point 있음)
--   waiting_tray      조제 계획까지 끝남, 조제 후 트레이 사진 대기 (hitl_point 없음)
--   done              처방 흐름 종료, 조제 기록 저장 완료
--   failed            오류로 중단
ALTER TABLE prescriptions DROP CONSTRAINT IF EXISTS prescriptions_status_check;
ALTER TABLE prescriptions ADD CONSTRAINT prescriptions_status_check
    CHECK (status IN ('processing', 'waiting_approval', 'waiting_tray', 'done', 'failed'));

-- 처방은 같은 ID의 대화에 속한다. conversations보다 먼저 만든 처방(11/12 이전 DB)은 대화를 채워 넣는다.
INSERT INTO conversations (id, pharmacy_id, user_id, created_at, updated_at)
SELECT id, pharmacy_id, user_id, created_at, updated_at FROM prescriptions
ON CONFLICT (id) DO NOTHING;
ALTER TABLE prescriptions DROP CONSTRAINT IF EXISTS prescriptions_conversation_fk;
ALTER TABLE prescriptions ADD CONSTRAINT prescriptions_conversation_fk
    FOREIGN KEY (id) REFERENCES conversations(id) ON DELETE CASCADE;

-- ------------------------------------------------------------ 이미지
CREATE TABLE IF NOT EXISTS images (
    id              UUID        PRIMARY KEY DEFAULT gen_random_uuid(),   -- A.1·A.2 image_id
    prescription_id UUID        NOT NULL REFERENCES prescriptions(id) ON DELETE CASCADE,
    kind            TEXT        CHECK (kind IN ('prescription', 'med_bag', 'package', 'tray')),   -- NULL = 판별 전
    kind_source     TEXT        CHECK (kind_source IN ('auto', 'manual')),
    storage_key     TEXT        NOT NULL,          -- 저장소 키 (1차: var/uploads 아래 상대 경로, 2차: 객체 저장소 키)
    width           INTEGER,                       -- 원본 크기(px)
    height          INTEGER,
    corners         JSONB,                         -- 트레이만: 꼭짓점 4점 [[x,y]×4] 원본 좌표, 보정 이미지 재생성용
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    CHECK (corners IS NULL OR kind = 'tray')
);
CREATE INDEX IF NOT EXISTS idx_images_rx ON images(prescription_id);

-- ------------------------------------------------------------ HITL 응답 이력 (추가만)
CREATE TABLE IF NOT EXISTS approvals (
    id              BIGINT      GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    prescription_id UUID        NOT NULL REFERENCES prescriptions(id) ON DELETE CASCADE,
    hitl_point      TEXT        NOT NULL CHECK (hitl_point IN ('type_select', 'rx_approval', 'inspection', 'risk')),
    action          TEXT        NOT NULL CHECK (action IN ('approve', 'edit', 'manual')),
    user_id         BIGINT      NOT NULL REFERENCES users(id),
    before_json     JSONB,                         -- interrupt 때 보여 준 내용
    after_json      JSONB,                         -- 약사가 보낸 내용
    reason          TEXT,                          -- manual이면 필수
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    CHECK (action <> 'manual' OR reason IS NOT NULL)
);
CREATE INDEX IF NOT EXISTS idx_approvals_rx ON approvals(prescription_id, created_at);

-- ------------------------------------------------------------ 트레이 검수 결과 (재촬영하면 여러 건, 추가만)
CREATE TABLE IF NOT EXISTS inspections (
    id              BIGINT      GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    prescription_id UUID        NOT NULL REFERENCES prescriptions(id) ON DELETE CASCADE,
    image_id        UUID        NOT NULL REFERENCES images(id),
    result_json     JSONB       NOT NULL,          -- A.3 전체
    hitl_required   BOOLEAN     NOT NULL,          -- A.3 summary.hitl_required
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_inspections_rx ON inspections(prescription_id, created_at);

-- ------------------------------------------------------------ 환자가 복용 중인 약
CREATE TABLE IF NOT EXISTS current_meds (
    id              BIGINT      GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    prescription_id UUID        NOT NULL REFERENCES prescriptions(id) ON DELETE CASCADE,
    item_code       TEXT        NOT NULL,          -- 품목기준코드
    name            TEXT        NOT NULL,
    source          TEXT        NOT NULL CHECK (source IN ('photo', 'manual')),
    image_id        UUID        REFERENCES images(id),
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (prescription_id, item_code),
    CHECK (source = 'manual' OR image_id IS NOT NULL)
);

-- ------------------------------------------------------------ 최종 조제 기록 (처방당 1건)
CREATE TABLE IF NOT EXISTS dispense_records (
    id              BIGINT      GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    prescription_id UUID        NOT NULL UNIQUE REFERENCES prescriptions(id) ON DELETE CASCADE,
    rx_json         JSONB       NOT NULL,          -- A.5 승인된 조제지시서
    plan_json       JSONB       NOT NULL,          -- A.4 조제 계획
    consult_json    JSONB,                         -- A.6 상담 결과
    risk_level      TEXT        CHECK (risk_level IN ('none', 'caution', 'warning', 'danger')),
    user_id         BIGINT      NOT NULL REFERENCES users(id),
    completed_at    TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_dispense_completed ON dispense_records(completed_at);

-- ------------------------------------------------------------ 대화 메시지 (타임라인, 추가만)
-- 약사 입력 1건(글 + 사진 1장) = 1행 = 슈퍼바이저 1회 실행. 시스템 응답은 card 행.
--   input : content = {"text": "..."} (글이 없으면 {}), image_id = 함께 올린 사진
--   card  : content = {"card": 종류, ...}. 종류: type_select · rx_approval · inspection · risk · answer · notice · error
--           원본이 다른 테이블에 있으면 참조 ID만 둔다 (approval_id, inspection_id).
--           상담 답변(answer)은 A.6을 content에 그대로 둔다 (상담 대화에는 dispense_records가 없다)
CREATE TABLE IF NOT EXISTS messages (
    id              UUID        PRIMARY KEY DEFAULT gen_random_uuid(),   -- Agent state의 message_id
    conversation_id UUID        NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
    sender          TEXT        NOT NULL CHECK (sender IN ('pharmacist', 'system')),
    kind            TEXT        NOT NULL CHECK (kind IN ('input', 'card')),
    content         JSONB       NOT NULL DEFAULT '{}'::jsonb,
    image_id        UUID        REFERENCES images(id),
    user_id         BIGINT      REFERENCES users(id),            -- input을 보낸 약사
    created_at      TIMESTAMPTZ NOT NULL DEFAULT clock_timestamp(),   -- 한 트랜잭션 안에서도 순서가 갈리게
    CHECK ((kind = 'input') = (sender = 'pharmacist')),
    CHECK (kind <> 'input' OR user_id IS NOT NULL),
    CHECK (kind <> 'input' OR coalesce(content->>'text', '') <> '' OR image_id IS NOT NULL),
    CHECK (kind <> 'card' OR content ? 'card'),
    CHECK (kind = 'input' OR image_id IS NULL)
);
CREATE INDEX IF NOT EXISTS idx_messages_conversation ON messages(conversation_id, created_at);

-- ------------------------------------------------------------ 트리거
-- prescriptions.updated_at 자동 갱신
CREATE OR REPLACE FUNCTION touch_updated_at() RETURNS trigger AS $$
BEGIN
    NEW.updated_at := now();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS trg_prescriptions_touch ON prescriptions;
CREATE TRIGGER trg_prescriptions_touch
    BEFORE UPDATE ON prescriptions
    FOR EACH ROW EXECUTE FUNCTION touch_updated_at();

DROP TRIGGER IF EXISTS trg_conversations_touch ON conversations;
CREATE TRIGGER trg_conversations_touch
    BEFORE UPDATE ON conversations
    FOR EACH ROW EXECUTE FUNCTION touch_updated_at();

-- 메시지가 추가되면 대화를 목록 맨 위로 (conversations.updated_at 갱신)
CREATE OR REPLACE FUNCTION touch_conversation() RETURNS trigger AS $$
BEGIN
    UPDATE conversations SET updated_at = now() WHERE id = NEW.conversation_id;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS trg_messages_touch_conversation ON messages;
CREATE TRIGGER trg_messages_touch_conversation
    AFTER INSERT ON messages
    FOR EACH ROW EXECUTE FUNCTION touch_conversation();

-- 이력 테이블은 수정 금지 (처방 삭제에 따른 CASCADE 삭제는 허용)
CREATE OR REPLACE FUNCTION forbid_update() RETURNS trigger AS $$
BEGIN
    RAISE EXCEPTION '% 는 이력 테이블이라 수정할 수 없습니다. 새 행을 추가하세요.', TG_TABLE_NAME;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS trg_approvals_no_update ON approvals;
CREATE TRIGGER trg_approvals_no_update
    BEFORE UPDATE ON approvals
    FOR EACH ROW EXECUTE FUNCTION forbid_update();

DROP TRIGGER IF EXISTS trg_inspections_no_update ON inspections;
CREATE TRIGGER trg_inspections_no_update
    BEFORE UPDATE ON inspections
    FOR EACH ROW EXECUTE FUNCTION forbid_update();

DROP TRIGGER IF EXISTS trg_messages_no_update ON messages;
CREATE TRIGGER trg_messages_no_update
    BEFORE UPDATE ON messages
    FOR EACH ROW EXECUTE FUNCTION forbid_update();
