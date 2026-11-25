-- 피지컬 AI 약국 조제 보조 서비스 — DB 스키마 (PostgreSQL 16)
--
-- 규칙
--   ID     : 모듈·JSON·파일 경로에 나가는 ID(처방, 이미지)는 UUID, 테이블 안에서만 쓰는 ID는 BIGINT IDENTITY
--   시각   : 모두 TIMESTAMPTZ (UTC로 저장, 화면에서 KST로 표시)
--   JSON   : 모듈 간 데이터(A.3~A.5)는 JSONB 원문 그대로, 필터에 쓰는 값만 컬럼으로 뽑는다
--   이력   : approvals, inspections는 추가만 한다 (UPDATE 금지 트리거)
--   처방   : 처방 1건 = LangGraph thread 1개 (thread_id = prescriptions.id)
--   체크포인트: LangGraph PostgresSaver.setup()이 같은 DB에 checkpoint* 테이블을 직접 만든다.
--             thread_id = prescriptions.id(UUID 문자열), FK 없이 값으로만 이어진다.
--
-- 앱 첫 실행 때 한 번 실행한다 (여러 번 실행해도 안전).

SET client_min_messages = warning;   -- 다시 실행할 때 "이미 있음" 안내를 숨긴다

-- ------------------------------------------------------------ 1차 범위 밖 정리 (11/25)
-- 처방전 단일 입력(명세서 v0.8)으로 빠진 대화·상담·복용 약·사진 종류 판별의 테이블과 컬럼.
-- 이미 만든 DB에서도 앱 시작(init-db) 때 지워지도록 둔다 (새 DB에서는 아무 일도 하지 않음).
DROP TABLE IF EXISTS messages;                 -- 트리거(trg_messages_*)도 함께 지워진다
DROP TABLE IF EXISTS current_meds;
ALTER TABLE IF EXISTS prescriptions DROP CONSTRAINT IF EXISTS prescriptions_conversation_fk;
DROP TABLE IF EXISTS conversations;            -- 트리거(trg_conversations_touch)도 함께 지워진다
DROP FUNCTION IF EXISTS touch_conversation();
ALTER TABLE IF EXISTS images DROP COLUMN IF EXISTS kind_source;
ALTER TABLE IF EXISTS dispense_records DROP COLUMN IF EXISTS consult_json;

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

-- ------------------------------------------------------------ 처방
CREATE TABLE IF NOT EXISTS prescriptions (
    id          UUID        PRIMARY KEY DEFAULT gen_random_uuid(),   -- LangGraph thread_id, A.3~A.5 prescription_id
    pharmacy_id BIGINT      NOT NULL REFERENCES pharmacies(id),
    display_no  TEXT        NOT NULL,                                -- 화면 표시용 번호 (예: P0001)
    user_id     BIGINT      NOT NULL REFERENCES users(id),
    status      TEXT        NOT NULL DEFAULT 'processing',          -- 값 목록은 아래 prescriptions_status_check
    hitl_point  TEXT        CHECK (hitl_point IN ('rx_approval', 'inspection')),
    error_code  TEXT,                                                -- 값 목록은 아래 prescriptions_error_code_check
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
--   cancelled         약사가 취소 (11/19 취소 기능에서 쓴다. 이력에는 완료건만 넘기므로 조제 기록은 없음)
ALTER TABLE prescriptions DROP CONSTRAINT IF EXISTS prescriptions_status_check;
ALTER TABLE prescriptions ADD CONSTRAINT prescriptions_status_check
    CHECK (status IN ('processing', 'waiting_approval', 'waiting_tray', 'done', 'failed', 'cancelled'));

-- 실패 코드 (명세서 A.7, 기능분석서 6장). E_INTERRUPTED는 처리 중에 서버가 다시 시작되어 끊긴 처방 (11/12 추가, 웹 전용)
ALTER TABLE prescriptions DROP CONSTRAINT IF EXISTS prescriptions_error_code_check;
ALTER TABLE prescriptions ADD CONSTRAINT prescriptions_error_code_check
    CHECK (error_code IN ('E_OCR_EMPTY', 'E_TRAY_NOT_FOUND', 'E_CV_INFER', 'E_PLAN_RANGE',
                          'E_LLM', 'E_EXT_API', 'E_DB', 'E_INTERRUPTED'));


-- ------------------------------------------------------------ 이미지
CREATE TABLE IF NOT EXISTS images (
    id              UUID        PRIMARY KEY DEFAULT gen_random_uuid(),   -- A.1·A.2 image_id
    prescription_id UUID        NOT NULL REFERENCES prescriptions(id) ON DELETE CASCADE,
    kind            TEXT,                          -- 값 목록은 아래 images_kind_check (웹이 항상 넣는다)
    storage_key     TEXT        NOT NULL,          -- 저장소 키 (1차: var/uploads 아래 상대 경로, 2차: 객체 저장소 키)
    width           INTEGER,                       -- 원본 크기(px)
    height          INTEGER,
    corners         JSONB,                         -- 트레이만: 꼭짓점 4점 [[x,y]×4] 원본 좌표, 보정 이미지 재생성용
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    CHECK (corners IS NULL OR kind = 'tray')
);
CREATE INDEX IF NOT EXISTS idx_images_rx ON images(prescription_id);

-- 사진 종류: prescription 처방전(메인 상단 업로드) · tray 트레이(트레이 대기 처방 카드 안 업로드). 올리는 자리로 구분한다
ALTER TABLE images DROP CONSTRAINT IF EXISTS images_kind_check;
ALTER TABLE images ADD CONSTRAINT images_kind_check CHECK (kind IN ('prescription', 'tray'));

-- ------------------------------------------------------------ HITL 응답 이력 (추가만)
CREATE TABLE IF NOT EXISTS approvals (
    id              BIGINT      GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    prescription_id UUID        NOT NULL REFERENCES prescriptions(id) ON DELETE CASCADE,
    hitl_point      TEXT        NOT NULL CHECK (hitl_point IN ('rx_approval', 'inspection')),
    action          TEXT        NOT NULL,          -- 값 목록은 아래 approvals_action_check
    user_id         BIGINT      NOT NULL REFERENCES users(id),
    before_json     JSONB,                         -- interrupt 때 보여 준 내용
    after_json      JSONB,                         -- 약사가 보낸 내용
    reason          TEXT,                          -- manual이면 필수
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    CHECK (action <> 'manual' OR reason IS NOT NULL)
);
CREATE INDEX IF NOT EXISTS idx_approvals_rx ON approvals(prescription_id, created_at);

-- 응답 종류 (기능분석서 F-WEB-07 · 09). 이미 만든 DB에도 앱 시작(init-db) 때 반영되도록 따로 건다 (11/18 HITL ② 추가)
--   rx_approval(HITL ①): approve 승인 · edit 수정 후 다시 계산 · manual 사유 입력 후 진행
--   inspection(HITL ②):  confirm 검수 확인 완료 · redo 재조제 (트레이를 다시 채워 사진 대기)
--                        manual 육안 검수 후 수동 진행 (11/23, 기능분석서 6장 E_CV_INFER: 검출 모델 오류로 자동 검수를
--                        못 할 때 사유를 적고 완료. 이때 조제 기록의 inspection_json은 비어 있다)
ALTER TABLE approvals DROP CONSTRAINT IF EXISTS approvals_action_check;
ALTER TABLE approvals ADD CONSTRAINT approvals_action_check
    CHECK (action IN ('approve', 'edit', 'manual', 'confirm', 'redo'));
ALTER TABLE approvals DROP CONSTRAINT IF EXISTS approvals_action_point_check;
ALTER TABLE approvals ADD CONSTRAINT approvals_action_point_check
    CHECK ((hitl_point = 'rx_approval' AND action IN ('approve', 'edit', 'manual'))
        OR (hitl_point = 'inspection' AND action IN ('confirm', 'redo', 'manual')));
-- 경고 확인 체크 (11/17 회의 Q11, 기능분석서 F-WEB-07 ④ · 5.3절). HITL ① 승인 때만 값이 있다 (확인할 경고가 없으면 false)
ALTER TABLE approvals ADD COLUMN IF NOT EXISTS warning_ack BOOLEAN;

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

-- ------------------------------------------------------------ 최종 조제 기록 (처방당 1건)
CREATE TABLE IF NOT EXISTS dispense_records (
    id              BIGINT      GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    prescription_id UUID        NOT NULL UNIQUE REFERENCES prescriptions(id) ON DELETE CASCADE,
    rx_json         JSONB       NOT NULL,          -- A.5 승인된 조제지시서
    plan_json       JSONB       NOT NULL,          -- A.4 조제 계획
    risk_level      TEXT        CHECK (risk_level IN ('none', 'caution', 'warning', 'danger')),   -- 조제지시서 risk_check.level
    user_id         BIGINT      NOT NULL REFERENCES users(id),   -- 마지막 HITL 응답자 (없으면 처방을 올린 사용자)
    completed_at    TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_dispense_completed ON dispense_records(completed_at);
-- 최종 검수 결과 A.3 (11/20 F-WEB-12 ②: 최종 조제지시서·조제 계획·검수 결과를 조제 기록으로 저장). 예전 DB에도 init-db로 반영
ALTER TABLE dispense_records ADD COLUMN IF NOT EXISTS inspection_json JSONB;

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
