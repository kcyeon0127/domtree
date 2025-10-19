# DOMTree Analyzer

사람이 인식하는 논리적 페이지 구조(Human Tree)와 LLM이 추출한 구조(LLM Tree)를 비교·분석하는 파이썬 도구입니다. 자동 스크린샷 수집, DOM 기반 트리 생성, 다양한 구조 비교 메트릭 계산, 배치 처리 및 시각화까지 한 번에 수행할 수 있습니다.

## 주요 기능
- Playwright 기반 자동 캡처: 쿠키 동의 팝업 닫기, JavaScript 렌더링 대기, 전체 페이지 스크롤 캡처 지원
- Human Tree 추출: 헤딩·시맨틱 태그·텍스트 밀도·레이아웃 키워드를 활용한 휴먼 퍼셉션 트리 구성
- LLM Tree 생성: 스크린샷 시각 특징(밝기, 주요 색상 등)과 추상화 파라미터를 반영한 휴리스틱 LLM 트리 제공 (실제 LLM 연동도 가능)
- 평가 지표: 트리 편집 거리(TED), 계층 F1, 구조적 유사도, 읽기 순서 정렬(Needleman-Wunsch), 불일치 패턴 분류
- 배치/리포트: URL 목록 일괄 처리, 요약 통계, CSV/JSON 결과물, 트리 시각화 이미지 출력

## 빠른 시작
```bash
conda activate domtree
pip install -r requirements.txt
pip install -e .
playwright install chromium
```
- 최초 1회 환경 생성이 필요하다면 `conda create -n domtree python=3.10`을 먼저 실행하세요.

## 사용 예시
### 단일 URL 분석
```bash
domtree analyze https://example.com
```
- 캡처 이미지와 렌더링된 HTML이 `data/screenshots/`에 저장됩니다.
- 분석 결과, 메트릭 요약, 트리 시각화는 `data/output/single/<슬러그>/<타임스탬프>/` 하위에 저장됩니다.
- CLI는 터미널에 메트릭을 출력하지 않으며, 모든 산출물을 파일로만 남깁니다.

### 오프라인 자산 분석
이미 수집한 `saved_page.html`, `saved_page.png`가 있을 경우:
```bash
domtree analyze-offline saved_page.html saved_page.png
```
- 필요하다면 세 번째 인자로 식별자(예: `domtree analyze-offline saved_page.html saved_page.png mylabel`)를 지정해 저장 폴더 이름을 제어할 수 있습니다.
- 결과물은 `data/output/offline/<식별자>/<타임스탬프>/`에 저장됩니다.

### 배치 처리
`urls.txt`(한 줄당 하나의 URL) 또는 `url` 컬럼을 가진 CSV/JSON 파일을 준비한 후:
```bash
domtree batch urls.txt
```
- 전체 요약(`summary.json`), 개별 결과(`results.json`), CSV(`results.csv`)가 `data/output/batch/<식별자>/<타임스탬프>/`에 저장됩니다.
- CLI는 터미널에 결과를 출력하지 않습니다.

## 구성 요소
- `domtree.capture`: Playwright 캡처 옵션과 전체 페이지 스크린샷/HTML 저장 로직
- `domtree.human_tree`: DOM 기반 휴먼 트리 추출 휴리스틱
- `domtree.llm_tree`: 휴리스틱 LLM 트리 생성기와 추상화 파라미터, 향후 실제 LLM 연동용 인터페이스
- `domtree.metrics`: TED, 계층 F1, 구조적 유사도, 읽기 순서 정렬, 불일치 패턴 분류 등 핵심 메트릭
- `domtree.pipeline`: 전체 파이프라인 오케스트레이션, 배치 처리, 시각화, 요약 통계
- `domtree.visualization`: 네트워크 그래프 기반 트리/비교 시각화
- `domtree.cli`: Typer CLI 커맨드 (`analyze`, `analyze-offline`, `batch`)

### CLI 기본 하이퍼파라미터 조정
터미널 실행 시에는 내부 기본값을 사용하며, 아래 상수를 수정해 하이퍼파라미터를 조절할 수 있습니다. (`--wait-after-load`와 같은 커맨드라인 옵션은 제공하지 않습니다.)

| 설정 사전 | 기본 키/값 | 설명 |
| --- | --- | --- |
| `_CAPTURE_SETTINGS` | `wait_after_load=1.0`, `max_scroll_steps=40` | 렌더링 대기 시간, 자동 스크롤 수행 횟수 |
| `_HUMAN_SETTINGS` | `min_text_length=25` | Human Tree에 포함할 최소 텍스트 길이 |
| `_LLM_SETTINGS` | `max_depth=4`, `max_children=6` | 휴리스틱 LLM Tree의 최대 깊이/자식 수 |

필요한 값을 `src/domtree/cli.py`에서 직접 수정한 뒤 CLI를 실행하면 변경사항이 즉시 적용됩니다.

## 확장 방법
- 실서비스용 LLM 연동: `LLMTreeGenerator` 추상 클래스를 구현해 스크린샷·HTML을 LLM에게 전달하고 트리를 받아오는 로직을 추가하세요.
- 맞춤 메트릭 추가: `domtree.metrics` 모듈에 신규 지표를 구현한 뒤 `domtree.comparison.compute_comparison`에 연결하면 됩니다.
- 대시보드/노트북 분석: `DomTreeAnalyzer` 클래스를 활용해 파이썬 스크립트나 노트북에서 직접 호출하고, CSV/JSON 결과를 후처리하세요.

## 출력 경로
- 캡처 산출물: `data/screenshots/`
- 배치/결과 리포트: 기본 `data/output/` (옵션으로 변경 가능)
- 시각화 이미지: `--visualize-dir` 또는 사용자 지정 경로

## 라이선스
프로젝트 라이선스는 별도 안내를 참고하거나 담당자에게 문의하세요.
