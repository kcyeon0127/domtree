# DOMTree Analyzer

사람이 인식하는 논리적 페이지 구조(Human Tree)와 LLM이 추출한 구조(LLM Tree)를 비교·분석하는 파이썬 도구입니다. 자동 스크린샷 수집, DOM 기반 트리 생성, 다양한 구조 비교 메트릭 계산, 배치 처리 및 시각화까지 한 번에 수행할 수 있습니다.

## 주요 기능
- Playwright 기반 자동 캡처: 쿠키 동의 팝업 닫기, JavaScript 렌더링 대기, 전체 페이지 스크롤 캡처 지원
- Human Tree 추출: 시각 Zone과 Heading 계층을 중첩한 휴먼 퍼셉션 트리 (읽기 순서, 역할, 시각 단서 포함)
- **뷰포트 필터링**: 스크린샷에 실제로 보이는 영역(초기 뷰포트)과 겹치는 요소만 트리에 포함하여 사람의 즉시 인식 범위에 집중
- LLM Tree 생성: 스크린샷 시각 특징(밝기, 주요 색상 등)과 추상화 파라미터를 반영한 휴리스틱 LLM 트리 제공 (실제 LLM 연동도 가능)
- 평가 지표: 트리 편집 거리(TED), 계층 F1, 구조적 유사도, 읽기 순서 정렬(Needleman-Wunsch), 불일치 패턴 분류
- 배치/리포트: URL 목록 일괄 처리, 요약 통계, CSV/JSON 결과물, 트리 시각화 이미지 출력

## 최근 변경 사항 (2025-10-21)
- Ollama Vision 연동 기본 엔드포인트를 `/api/generate`로 통일해 400 오류를 방지했습니다 (`src/domtree/cli.py`, `README` 예제).
- README에 Ollama Vision 모델 사용 시 `/api/chat`이 아닌 `/api/generate`를 사용해야 한다는 주석을 추가했습니다.
- 실행 결과물이 자동으로 저장되는 기본 경로 설명을 보강해 `--visualize-dir`, `--output-json` 없이도 PNG/JSON이 생성된다는 점을 명확히 했습니다.
- Ollama Vision 응답이 서술형 텍스트로 반환되더라도 JSON 블록을 추출해 파싱하도록 보강하고, `format="json"` 옵션으로 JSON 출력 준수를 강제했습니다.
- JSON 파싱에 실패하더라도 `llm_error` 노드를 반환해 파이프라인이 중단되지 않고, `llm_tree.json`에 원본 응답(`attributes.llm.raw_response`)과 요약이 기록되도록 했습니다.
- 템플릿/스키마 응답을 자동 감지해 최대 3회까지 재시도하며, 각 재시도 프롬프트에 교정 메시지를 삽입하고 실패 사유·프롬프트 해시·시도 횟수를 메타데이터로 남기도록 강화했습니다.
- JSON Schema(Draft-07)를 정식 정의해 LLM 응답을 검증하며, 스키마 본문을 프롬프트에 포함해 “해당 스키마를 따르라”고 명시했습니다.

## 빠른 시작
```bash
conda activate domtree
pip install -r requirements.txt
pip install -e .
playwright install chromium
playwright install-deps
```
- 최초 1회 환경 생성이 필요하다면 `conda create -n domtree python=3.10`을 먼저 실행하세요.
- (Linux 환경) 브라우저 실행 의존성: `playwright install-deps`
- LLaMA 3.2 Vision 사용 시, 별도 터미널에서 Ollama 서버 실행: `ollama serve` (기본 주소 `http://localhost:11434`, 모델은 `ollama pull llama3.2-vision:11b`로 사전 다운로드)

## 사용 예시
### 단일 URL 분석
```bash
domtree analyze https://example.com

domtree analyze https://ko.wikipedia.org/wiki/%ED%8C%8C%EC%9D%B4%EC%8D%AC
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
- `domtree.human_tree`: **Zone(시각 영역) ⊃ Heading(논리 계층)** 중첩 트리를 생성하는 휴먼 퍼셉션 추출기
- `domtree.llm_tree`: 휴리스틱 LLM 트리 생성기 + Ollama 기반 LLaMA 3.2 Vision 11B 연동 구현 (`OllamaVisionLLMTreeGenerator`), 기타 LLM 연동을 위한 인터페이스
- `domtree.metrics`: TED, 계층 F1, 구조적 유사도, 읽기 순서 정렬, 불일치 패턴 분류 등 핵심 메트릭
- `domtree.pipeline`: 전체 파이프라인 오케스트레이션, 배치 처리, 시각화, 요약 통계
- `domtree.visualization`: 네트워크 그래프 기반 트리/비교 시각화
- `domtree.cli`: Typer CLI 커맨드 (`analyze`, `analyze-offline`, `batch`)

### 사람 인식 트리 스키마
`HumanTreeExtractor`는 Zone Tree와 Heading Tree를 중첩한 구조를 생성합니다. 각 노드는 `metadata`에 다음과 같은 공통 필드를 유지합니다.

| 필드 | 설명 |
| --- | --- |
| `type` | `page`, `zone`, `section`, `paragraph`, `list`, `table`, `figure` 등 노드 유형 |
| `role` | `main`, `sidebar`, `nav`, `toc`, `ad`, `body` 등 시각 영역 역할 |
| `text_heading`/`heading_level` | 헤딩 텍스트와 등급(H1~H6) |
| `reading_order` | 사람이 읽는 순서를 기준으로 한 순번 (좌→우, 상→하 원칙) |
| `dom_refs` | DOM 요소를 추적하기 위한 CSS 선택자(`#id`, `.class` 등) |
| `vis_cues` | 인라인 스타일 기반 시각 단서(`font_size`, `font_weight`, `margin_top` 등) |
| `text_preview` | 본문/리스트 항목 등 텍스트 블록의 미리보기 |
| `notes` | 휴리스틱·LLM 정보 등 추가 메모 (예: `notes.llm.confidence`) |

예시(JSON):

```json
{
  "name": "zone",
  "label": "Main",
  "metadata": {
    "type": "zone",
    "role": "main",
    "reading_order": 1,
    "dom_refs": ["main.content"],
    "text_heading": "문서 제목"
  },
  "children": [
    {
      "name": "section",
      "label": "소개",
      "metadata": {
        "type": "section",
        "heading_level": 2,
        "text_heading": "소개",
        "reading_order": 2
      },
      "children": [
        {
          "name": "paragraph",
          "label": "소개 본문 미리보기",
          "metadata": {
            "type": "paragraph",
            "text_preview": "...",
            "reading_order": 3
          }
        }
      ]
    }
  ]
}
```

> 기본 동작은 스크린샷 뷰포트에 실제로 렌더링된 요소만 유지합니다. Playwright 캡처 단계에서 모든 노드에 `data-domtree-bbox`(bounding box)와 `data-domtree-viewport`(뷰포트 크기) 속성을 주입하고, `HumanTreeExtractor`가 이를 사용해 뷰포트와 겹치지 않는 노드를 제외합니다. 필요하면 `HumanTreeOptions(restrict_to_viewport=False)`로 전체 DOM을 분석하도록 변경할 수 있습니다.

## Human Tree/Human vs LLM 워크플로
1. **Capture**: Playwright가 렌더링 완료까지 대기한 뒤 전체 페이지를 스크롤·캡처하고 HTML을 저장합니다. 이때 모든 DOM 요소에 `data-domtree-bbox`/`data-domtree-viewport` 속성을 주입해 뷰포트 좌표를 기록합니다.
2. **Human Tree 추출**: Zone(시맨틱 컨테이너) → Heading → 콘텐츠 블록 순으로 계층을 구축하고, `reading_order`, `dom_refs`, `vis_cues`(bbox 포함) 등 메타데이터를 채웁니다. 기본 설정(`restrict_to_viewport=True`)은 스크린샷에 실제 보인 부분만 유지합니다.
3. **LLM Tree 생성**: 기본 구현은 Human Tree를 요약해 LLM이 인지할 법한 구조를 휴리스틱으로 근사합니다. 실제 모델 연동 시 `LLMTreeGenerator`를 구현하거나 파라미터를 조정해 자유롭게 구성할 수 있습니다.
4. **비교 및 평가**: TED, Hierarchical F1, Structural Similarity, Reading Order Alignment, mismatch 리포트로 두 트리의 차이를 정량화합니다.
5. **결과 산출**: `data/output/<mode>/<slug>/<timestamp>/`에 JSON(트리/메트릭)과 PNG(사람/LLM 비교)를 기록하고, `human_tree.json`/`llm_tree.json`에서 모든 텍스트와 메타데이터를 확인할 수 있습니다.

## LLM 비교 및 평가 지표 (실제 구현 기준)
- **Tree Edit Distance / Normalized TED** (`domtree.metrics.ted`): `zss`를 이용해 삽입·삭제·치환 비용으로 구조 차이를 계산합니다.
- **Hierarchical Precision/Recall/F1** (`domtree.metrics.hierarchical_f1`): 루트→노드 경로 집합을 비교하여 정밀도/재현율/조화평균을 산출합니다.
- **Structural Similarity** (`domtree.metrics.structure`): 노드 수, 깊이, 평균 branching, 라벨 분포(코사인 유사도)의 차이를 정규화된 점수로 제공합니다.
- **Reading Order Alignment** (`domtree.metrics.reading_order`): Needleman–Wunsch 동적 계획법으로 읽기 순서 정렬 점수와 갭(누락된 순서)을 구합니다.
- **Mismatch Patterns** (`domtree.metrics.mismatch`): 누락/추가 노드 수, 깊이 이동(Depth Shift), 읽기 순서 갭 등 불일치 유형을 요약합니다.

시각화 모듈은 `_FONT_CANDIDATES` 순서대로 사용 가능한 폰트를 탐색하여 `font.family` 스택을 구성하므로, 앞쪽에 있는 범용 폰트(Noto Sans CJK, Arial Unicode 등)가 설치되어 있으면 한글/한자/라틴 혼용 텍스트도 자동으로 커버됩니다. 필요 시 `src/domtree/visualization/tree_plot.py`의 목록(예: `"Nanum Gothic"` → 시스템에 따라 이름이 다를 수 있음)을 조정하거나, 아래 명령으로 폰트를 설치하세요.

- macOS(Homebrew): `brew install --cask nanumfont` 또는 `brew install --cask noto-sans-cjk-kr`
- Ubuntu/Debian: `sudo apt install fonts-nanum fonts-noto-cjk`
- Conda 환경: `conda install -c conda-forge noto-fonts-cjk` (macOS arm64에서는 일부 폰트 패키지가 제공되지 않을 수 있습니다)
- 번들 사용: `src/assets/fonts/`에 원하는 `.ttf/.otf/.ttc` 파일을 넣으면 앱이 자동으로 모두 로드합니다. 예: `Nanum Gothic`, `Noto Sans CJK`, `Noto Sans Sinhala` 등을 넣어두면 서버 환경에서도 동일한 폰트 스택이 적용됩니다. 추가 폰트를 넣었다면 라이선스 파일도 함께 보관하세요.
- 시각화 PNG에서는 가독성을 위해 지원되지 않는 문자(폰트에 없는 스크립트)는 자동으로 제거/생략됩니다. 원본 텍스트는 `human_tree.json`/`llm_tree.json` 메타데이터에 그대로 남습니다.

### CLI 기본 하이퍼파라미터 조정
터미널 실행 시에는 내부 기본값을 사용하며, 아래 상수를 수정해 하이퍼파라미터를 조절할 수 있습니다.

| 설정 사전 | 기본 키/값 | 설명 |
| --- | --- | --- |
| `_CAPTURE_SETTINGS` | `wait_after_load=1.0`, `max_scroll_steps=40` | 렌더링 대기 시간, 자동 스크롤 수행 횟수 |
| `_HUMAN_SETTINGS` | `min_text_length=20`, `restrict_to_viewport=True` | Human Tree에 포함할 최소 텍스트 길이, 뷰포트 필터링 여부 |
| `_LLM_SETTINGS` | `max_depth=4`, `max_children=6` | 휴리스틱 LLM Tree의 최대 깊이/자식 수 |
| `_LLM_BACKEND`, `_OLLAMA_ENDPOINT`, `_OLLAMA_MODEL` | 기본 LLM 백엔드(`"ollama"`), Ollama API 주소, 모델 식별자 | CLI 옵션 없이 내부 상수를 수정해 백엔드를 전환 |

필요한 값을 `src/domtree/cli.py`에서 직접 수정한 뒤 CLI를 실행하면 변경사항이 즉시 적용됩니다.

## 확장 방법
- 실서비스용 LLM 연동: `LLMTreeGenerator` 추상 클래스를 구현해 스크린샷·HTML을 LLM에게 전달하고 트리를 받아오는 로직을 추가하세요.
- 맞춤 메트릭 추가: `domtree.metrics` 모듈에 신규 지표를 구현한 뒤 `domtree.comparison.compute_comparison`에 연결하면 됩니다.
- 대시보드/노트북 분석: `DomTreeAnalyzer` 클래스를 활용해 파이썬 스크립트나 노트북에서 직접 호출하고, CSV/JSON 결과를 후처리하세요.

### 예: Ollama + LLaMA 3.2 Vision 11B 연동

1. **서버 준비**
   ```bash
   # Ollama 설치 (macOS)
   brew install ollama

   # 모델 다운로드 및 서버 실행
   ollama pull llama3.2-vision:11b
   ollama serve
   ```

2. **파이프라인에서 사용**
   ```python
   from domtree.pipeline import DomTreeAnalyzer
   from domtree.llm_tree import OllamaVisionLLMTreeGenerator, OllamaVisionOptions

   options = OllamaVisionOptions(
       endpoint="http://localhost:11434/api/generate",
       model="llama3.2-vision:11b",
       max_retries=3,
       # 필요 시 response_format=None 으로 해제 가능
       # response_format="json" 이면 Ollama가 JSON 출력을 보장하려 시도
    )

   analyzer = DomTreeAnalyzer(
       llm_generator=OllamaVisionLLMTreeGenerator(options=options)
   )

   result = analyzer.analyze_url("https://ko.wikipedia.org/wiki/파이썬")
   print(result.metrics)
   ```

   - 프롬프트는 스크린샷만 기반으로 구조를 추론하도록 구성되어 있습니다(HTML은 전달하지 않습니다).
   - 스크린샷은 자동으로 base64로 인코딩되어 메시지 컨텐츠로 전달됩니다.
   - 기본적으로 `format="json"`이 설정되어, 모델이 JSON만 반환하도록 강제합니다. 그래도 서술형 텍스트가 섞여 나오면 코드가 자동으로 JSON 블록을 추출해 파싱합니다.
   - 템플릿(예: `"zone|section|..."`, `"optional heading"`)을 감지하면 자동으로 재시도하며, JSON Schema 검증에서 누락/형식 오류가 나면 교정 지침을 추가해 다시 요청합니다.
   - JSON을 끝내 파싱하지 못하면 `llm_error` 루트 노드로 대체하여 분석이 계속되고, `llm_tree.json` → `attributes.llm.raw_response`에 원본 응답과 프롬프트 해시가 그대로 저장됩니다.
- Ollama Vision 모델은 `/api/generate` 엔드포인트를 사용해야 하며, `/api/chat`은 메시지 배열 형식만 지원하므로 400 오류가 발생합니다.

### LLM 응답 안정화 전략
- **강제 JSON 모드**: `format="json"`을 기본 활성화해 모델이 JSON만 돌려주도록 요청합니다.
- **네거티브 가드**: 시스템 규칙에 “구체 값만, 스키마/코드펜스 금지” 문구를 강하게 삽입했습니다.
- **템플릿 감지 & 교정 프롬프트**: `zone|section|…`, `optional heading` 등 마커가 감지되면 “스키마가 아닌 실제 데이터를 출력하라”는 교정 메시지를 추가해 재시도합니다.
- **간이 스키마 검증**: `name`, `metadata.type`, `children`이 비어 있거나 플레이스홀더가 포함되면 실패로 간주하고 교정 후 재요청합니다.
- **정식 JSON Schema 검증**: `src/domtree/schema.py`에 정의한 `TREE_JSON_SCHEMA`(Draft-07)를 사용해 응답을 검증하고, 스키마 위반 시 구체적인 오류 메시지로 재시도 지침을 전달합니다.
- **재시도 메타로그**: 최대 시도 횟수(`max_retries`, 기본 3)와 최종 프롬프트 해시, 원문 응답을 `notes.llm`/`attributes.llm`에 기록해 디버깅을 돕습니다.
- `max_retries`나 `template_markers`는 `OllamaVisionOptions` 인자로 조정할 수 있습니다.
- **커스터마이징 예시**:
  ```python
  options = OllamaVisionOptions(
      max_retries=5,
      template_markers=(
          "zone|section|paragraph|list|table|figure",
          "optional heading",
          "placeholder",
      ),
  )
  ```

### JSON Schema 개요
LLM이 따라야 하는 JSON Schema는 `src/domtree/schema.py`의 `TREE_JSON_SCHEMA`에 정의되어 있으며, 루트 노드/자식 노드/`metadata`/`vis_cues` 필드 구조를 Draft-07 형식으로 명시합니다. 이 스키마 전문은 LLM 프롬프트에도 그대로 포함되어 있어, 모델이 출력 형식을 벗어나지 않도록 강제합니다.

3. **CLI에서 사용**

   ```bash
   domtree analyze https://ko.wikipedia.org/wiki/파이썬
   ```

   - 기본 동작은 `_LLM_BACKEND = "ollama"` 값에 따라 Ollama Vision 모델을 사용합니다.
   - Heuristic LLM 트리로 테스트하고 싶다면 `src/domtree/cli.py` 상단의 `_LLM_BACKEND` 값을 `"heuristic"`으로 수정한 뒤 실행하세요.
   - Ollama 서버 주소나 모델을 바꾸려면 `_OLLAMA_ENDPOINT`, `_OLLAMA_MODEL` 상수를 조정하면 됩니다.

### 향후 ChatGPT API 연동 시

- `LLMTreeGenerator`를 상속하는 `ChatGPTLLMTreeGenerator`(예시 이름)를 작성해 OpenAI API 호출 코드를 구현합니다.
- 프롬프트는 README에 정의된 JSON 스키마를 그대로 요구해야 하며, 응답을 `TreeNode.from_dict()`로 역직렬화합니다.
- CLI에서 사용할 때는 `_LLM_BACKEND`를 `"chatgpt"`로 새로 정의하고, 그에 맞는 제너레이터를 `_create_llm_generator()`에 추가하면 전환할 수 있습니다.

## 출력 경로
- 캡처 산출물: `data/screenshots/`
- 배치/결과 리포트: 기본 `data/output/` (옵션으로 변경 가능)
- 시각화 이미지: 각 실행 디렉터리(`comparison.png`, `human.png`, `llm.png`)에 자동 저장

## 라이선스
프로젝트 라이선스는 별도 안내를 참고하거나 담당자에게 문의하세요.
