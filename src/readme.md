<!-- readme_src.md -->

# Documentação Técnica (Pipeline A → B → C)

Este documento descreve o funcionamento interno do pipeline, com **métricas**, **fórmulas** e **decisões de implementação** — com foco especial nas **heurísticas e otimizações do Step A**.

---

## 1) Fluxo geral do pipeline

### A) Step A — Faces + Identidade + Emoções (`step_a_faces_emotions.py`)
Entrada: `data/input.mp4`  
Saídas:
- `outputs/stepA_annotated.mp4`
- `outputs/stepA_summary.txt`

Processo (alto nível):
1. Leitura de frames (OpenCV) com **I/O em thread** (FileVideoStream)
2. **Amostragem temporal** via `FRAME_STEP`
3. Detecção de faces (DeepFace.extract_faces) em **frame reduzido** (`SCALE_DETECCAO`)
4. Reprojeção de bounding box para o frame original
5. **Heurísticas de filtragem** (área, aspect ratio, confiança e persistência temporal)
6. Recorte do rosto com margem (`PAD_RATIO`)
7. Emoção (DeepFace.analyze) com **skip + fallback**
8. Identidade (face_recognition) via distância e limiar
9. Anotação no frame + resumo

---

### B) Step B — Atividades / Pose (`step_b_activities.py`)
Entrada típica: vídeo do Step A (ou o original, conforme config do projeto)  
Saídas:
- `outputs/stepB_annotated.mp4`
- `outputs/stepB_summary.txt`

Processo:
1. Leitura frame a frame
2. Pose (MediaPipe)
3. Heurísticas para classificar atividades
4. Contagem em **frames por atividade**
5. Anotação no frame + resumo

---

### C) Step C — Consolidação (`step_c_summary.py`)
Entrada:
- `outputs/stepA_summary.txt`
- `outputs/stepB_summary.txt`

Saída:
- `outputs/relatorio_final.txt`

Objetivo:
- Consolidar resultados
- Inserir um **📊 CONTEXTO DE PROCESSAMENTO** para evitar interpretações erradas (ex.: somar frames do Step A e B)

---

## 2) Configurações relevantes (Step A)

As configs ficam em `criar_config()` (nomes típicos):

- `VIDEO_ENTRADA`: `data/input.mp4`
- `PASTA_FACES_CONHECIDAS`: `data/known_faces`
- `VIDEO_SAIDA`: `outputs/stepA_annotated.mp4`
- `RESUMO_SAIDA`: `outputs/stepA_summary.txt`

### Performance
- `FRAME_STEP` (ex.: 3)
- `SCALE_DETECCAO` (ex.: 0.7)
- `align=False` na detecção (mais rápido)
- Leitura em thread com fila (reduz gargalo de I/O)

### Robustez
- `DETECTOR_BACKEND = "opencv"`
- `ENFORCE_DETECTION = False`
- Warm-up + autoajuste de limiares (percentis)
- Filtros: área, AR (aspect ratio), confiança
- Persistência temporal (`K_PERSISTENCIA`, `TAMANHO_GRID`)
- `PAD_RATIO` para recorte de rosto com margem
- Emoção com fallback

---

## 3) Heurísticas e Otimizações do Step A (com fórmulas)

Esta seção detalha o “porquê” do Step A ser **rápido e robusto**, e como cada heurística é calculada.

---

### 3.1) Leitura otimizada de vídeo (I/O em thread)

**Problema:** leitura de frames pode virar gargalo (I/O), especialmente quando o processamento do frame é pesado (DeepFace + face_recognition).

**Solução:** `FileVideoStream` lê frames em uma thread e guarda em uma fila (`queue`).  
- Benefício: computação e I/O ficam desacoplados, reduzindo “paradas” na CPU/loop principal.
- Parâmetros típicos:
  - `queue_size = 128` (tamanho da fila)
  - thread daemon

**Modelo mental:**
- Thread A: lê `frame_t` e enfileira
- Thread B (main): consome `frame_t` e processa

---

### 3.2) Amostragem temporal (`FRAME_STEP`)

**Objetivo:** reduzir custo total processando apenas parte dos frames.

Se:
- `F_total` = total de frames do vídeo
- `step` = `FRAME_STEP`

Então uma aproximação é:

- **F_analisados_A ≈ ceil(F_total / step)**

Exemplo:
- `F_total = 3951`, `step = 3`
- `F_analisados_A ≈ ceil(3951/3) = 1317`

**Trade-off:**
- ✅ Reduz custo ~ proporcional a `step`
- ⚠️ Pode perder eventos muito rápidos entre frames amostrados

---

### 3.3) Downscale na detecção (`SCALE_DETECCAO`) + reprojeção da bbox

**Objetivo:** detecção de faces é cara porque roda em cima de muitos pixels.

Se:
- `S` = `SCALE_DETECCAO` (ex.: 0.7)
- `W, H` = dimensões do frame original
- `W' = S·W`, `H' = S·H` = dimensões do frame reduzido

Então:
- **Pixels reduzidos = (W'·H') = (S² · W·H)**

Ou seja, o custo de detecção tende a cair aproximadamente com **S²**.

Exemplo com `S=0.7`:
- `S² = 0.49`  
- Processa ~49% dos pixels → ganho típico ~2x (aprox.) na etapa de detecção.

#### Reprojeção de bbox (do frame pequeno para o original)

A detecção retorna uma bbox no frame reduzido:
- `(x_s, y_s, w_s, h_s)`

Para desenhar e recortar corretamente no frame original:
- **x = floor(x_s / S)**
- **y = floor(y_s / S)**
- **w = floor(w_s / S)**
- **h = floor(h_s / S)**

Depois é aplicado **clamp** para manter bbox dentro da imagem:
- `x ∈ [0, W-1]`
- `y ∈ [0, H-1]`
- `w ∈ [1, W-x]`
- `h ∈ [1, H-y]`

---

### 3.4) Warm-up + autoajuste estatístico de limiares (percentis)

**Problema:** valores de área da face, aspect ratio e confiança variam muito de vídeo para vídeo.

**Solução:** no início, coletar amostras (warm-up) e definir limiares por percentis.

#### Coleta de amostras (warm-up)
Durante o warm-up, para cada detecção válida:
- calcula `area = w·h`
- calcula `AR = w/h`
- armazena `confidence` quando existir

**Filtro de outliers no warm-up:**
- ignora faces muito grandes:
  - **area < 0.6 · area_frame**

onde:
- `area_frame = W·H` do vídeo

#### Definição dos limiares por percentis
Depois de acumular N amostras (ex.: `FRAMES_WARMUP_ANALISADOS = 150`), define:

- **MIN_AREA_FACE = P10(area)**
- **MAX_AREA_FACE = P95(area)**
- **MIN_AR = P5(AR)**
- **MAX_AR = P95(AR)**
- **MIN_CONFIANCA = P20(confidence)** (se houver amostras), senão 0.0

onde `Pk` é o k-ésimo percentil.

**Fallback do AR (robustez):**  
Se o intervalo ficar “apertado demais”:
- se `(MAX_AR - MIN_AR) < 0.15`, então:
  - `MIN_AR = 0.6`
  - `MAX_AR = 1.6`

**Motivo:** evita rejeitar tudo quando o percentil colou por distribuição ruim/curta.

---

### 3.5) Filtros geométricos (área e aspect ratio)

Após definir/usar limiares, uma detecção só passa se:

- **MIN_AREA_FACE ≤ area ≤ MAX_AREA_FACE**
- **MIN_AR ≤ AR ≤ MAX_AR**

Onde:
- `area = w·h`
- `AR = w/h`

**Efeito:** reduz falsos positivos:
- “quadradinhos” muito pequenos
- bboxes muito esticadas (AR fora do padrão de face)

---

### 3.6) Filtro de confiança do detector

Se o backend retornar `confidence`:
- a bbox só passa se:
  - **confidence ≥ MIN_CONFIANCA**

Se **não houver confidence** (ou vier None):
- a detecção é aceita (não penaliza backends que não fornecem score).

**Efeito:** reduz falsos positivos em cenas difíceis.

---

### 3.7) Persistência temporal por grid (reduzir “piscadas”)

**Problema:** detecções instáveis podem “piscando” (aparecem em 1 frame e somem no seguinte).

**Solução:** só aceitar uma face se ela persistir por `K` ocorrências recentes dentro de uma mesma região (grid).

Define-se um ID aproximado para a face:
- **id_face = (round(x / grid), round(y / grid))**
onde:
- `grid = TAMANHO_GRID` (ex.: 60)

Mantém-se um histórico (deque):
- `historico_ids` com `maxlen` (ex.: 10)

Critério de aceitação:
- se `K <= 1`: passa sempre
- senão:
  - **count(historico_ids == id_face) ≥ K**

**Interpretação:**
- A face precisa aparecer “no mesmo quadrante” pelo menos `K` vezes no histórico recente.

**Efeito prático:**
- ✅ reduz falsos positivos intermitentes
- ⚠️ pode atrasar a primeira aparição em ~K amostras (trade-off)

---

### 3.8) Recorte do rosto com padding (PAD_RATIO)

Para melhorar emoção/identidade, o recorte inclui margem ao redor da face.

Se:
- bbox = `(x, y, w, h)`
- `pad_ratio` = `PAD_RATIO`

Então:
- **pad = pad_ratio · max(w, h)**

E o recorte vira:
- `x1 = max(0, x - pad)`
- `y1 = max(0, y - pad)`
- `x2 = min(W, x + w + pad)`
- `y2 = min(H, y + h + pad)`

**Efeito:**
- reduz cortes “apertados” que atrapalham emoção
- dá mais contexto para landmarks (mesmo com align=False)

---

### 3.9) Emoção (DeepFace) — estratégia “skip + fallback”

**Objetivo:** reduzir custo evitando redetecção dentro do `DeepFace.analyze`.

Estratégia:
1) tenta:
   - `detector_backend="skip"`
   - `enforce_detection=False`
2) se falhar, fallback:
   - `DeepFace.analyze` padrão com `enforce_detection=False`

**Correção adicional (robustez):**
- se o crop for muito pequeno (ex.: <48×48), redimensiona para 96×96 antes da análise.

**Efeito:**
- ✅ acelera quando o crop já é confiável
- ✅ mantém robustez quando o skip falha

---

### 3.10) Identidade (face_recognition) — distância e limiar

Pipeline:
1. converte `crop` para RGB
2. calcula encoding no crop inteiro (localização conhecida)
3. calcula distância para a base de encodings conhecidos:
   - `dist_i = face_distance(known_encodings, enc_atual)`
4. escolhe o menor:
   - `i* = argmin(dist_i)`

Regra:
- se **dist(i*) < 0.55**, então:
  - identidade = `known_names[i*]`
- senão:
  - identidade = `"Desconhecido"`

**Efeito:**
- limiar menor → menos falsos positivos (mais conservador)
- limiar maior → mais matches (maior risco de confundir)

---

## 4) Fórmulas e Métricas (resumo)

### 4.1) Frames analisados no Step A (amostragem)
- **F_analisados_A ≈ ceil(F_total / FRAME_STEP)**

### 4.2) Frames analisados no Step B (frame a frame)
- **F_analisados_B = F_total**

### 4.3) Redução de pixels na detecção por downscale
- **pixels_small = S² · pixels_original**
- ganho típico ~ proporcional a `1/S²`

### 4.4) Contagem de atividades (Step B)
Para cada atividade `k`:
- **count(k) = Σ I(atividade_t = k)**, para `t = 1..F_total`

### 4.5) Contagem de emoções (Step A)
No Step A, em frames amostrados e faces válidas:
- **count(emocao e) = Σ Σ I(emocao(face_i, frame_t) = e)**

---

## 5) Contexto de Processamento (relatório final)

O Step C inclui:

```text
📊 CONTEXTO DE PROCESSAMENTO

- Total de frames do vídeo: <F_total>

Step A — Emoções e Faces:
- Frames analisados: <F_analisados_A>
- Estratégia: amostragem temporal (1 a cada <FRAME_STEP> frames)

Step B — Atividades Corporais:
- Frames analisados: <F_analisados_B>
- Estratégia: análise frame a frame
