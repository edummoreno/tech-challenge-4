# Pipeline de Processamento (Detalhes Técnicos)

Este documento descreve as heurísticas, fórmulas matemáticas e otimizações de **engenharia de software** e **visão computacional** implementadas nos scripts `src/`.

---

# Engenharia de Performance (Otimizações)

Para garantir o processamento próximo ao tempo real (*Real-Time Inference*), implementamos duas otimizações críticas na camada de I/O e no motor de inferência.

## 1. Leitura Assíncrona (I/O Threading)

A leitura de frames via `cv2.VideoCapture.read()` é uma operação bloqueante (I/O Bound). No modelo sequencial padrão, a GPU/CPU fica ociosa enquanto o disco lê o próximo frame.

**Implementação:**
Utilizamos o padrão **Producer-Consumer**. Uma *Thread* dedicada lê os frames e os coloca em uma fila (Buffer/Queue), enquanto a *Thread Principal* consome esses frames para processamento.

**Ganho Teórico:**
Seja $t_{read}$ o tempo de leitura e $t_{proc}$ o tempo de processamento (inferência).

- **Sem Threading (Sequencial):**
  $$
  T_{total} = \sum_{i=0}^{N} (t_{read} + t_{proc})
  $$

- **Com Threading (Paralelo):**
  Como as operações ocorrem simultaneamente, o tempo por frame tende ao maior dos dois tempos (gargalo), e não à soma:
  $$
  T_{total} \approx \sum_{i=0}^{N} \max(t_{read}, t_{proc})
  $$

Isso elimina o *overhead* de I/O, permitindo que a IA utilize 100% dos recursos computacionais disponíveis.

## 2. Backend de Inferência (MediaPipe vs OpenCV)

Substituímos o detector padrão `opencv` (Haar Cascades) pelo backend `mediapipe` (Google BlazeFace) dentro do wrapper `DeepFace`.

- **Haar Cascades:** Baseado em *features* artesanais e janelas deslizantes. Sensível à iluminação e rotação.
- **BlazeFace (MediaPipe):** Rede Neural Leve (Lightweight CNN) desenhada para inferência em CPU móvel.

**Vantagem:** O BlazeFace possui uma arquitetura SSD (Single Shot Detector) otimizada, oferecendo maior robustez a oclusões parciais e ângulos de face, com latência de inferência ($L_{inf}$) significativamente menor em CPUs:

$$
L_{inf}(MediaPipe) \ll L_{inf}(Haar) \quad \text{para resoluções } > 480p
$$

---

# Passo A — Reconhecimento facial + Emoções (com robustez)

Este passo implementa:

- **Reconhecimento facial:** Identificação biométrica comparando *encodings* (vetores de 128 dimensões).
- **Análise de emoções:** Classificação de expressões faciais.

Além das otimizações acima, mantivemos os **3 Patches de Robustez** para filtrar ruído:

## Visão geral do pipeline

O vídeo é lido via *buffer*, mas a análise “pesada” (DeepFace) segue uma amostragem temporal para economizar recursos:

$$
frame\_idx \bmod N = 0
$$

Onde $N$ é o `FRAME_STEP`.

---

## Patch 1: Autoajuste de Limiares (Warm-up)

Durante os primeiros `FRAMES_WARMUP` frames, coletamos estatísticas das detecções para definir o tamanho mínimo e máximo de um rosto aceitável naquele cenário específico.

Calculamos os limiares dinamicamente usando **percentis**:

$$
MinArea = P_{10}(AmostrasArea) \\
MaxArea = P_{95}(AmostrasArea)
$$

---

## Patch 2: Filtro por Confiança

Definimos um limiar de confiança $C_{min}$. Se a face detectada pela rede neural tiver score $C < C_{min}$, é descartada como falso positivo.

$$
Face_{valida} \iff Confidence(Face) \ge C_{min}
$$

---

## Patch 3: Persistência Temporal

Para evitar o efeito de "flickering" (rosto piscando na tela), exigimos consistência temporal.

1. Discretizamos a posição do centro da face $(c_x, c_y)$ em um **Grid** de tamanho $G$.
2. Geramos um ID espacial aproximado:
   $$
   ID_{aprox} = \left( \text{round}\left(\frac{c_x}{G}\right), \text{round}\left(\frac{c_y}{G}\right) \right)
   $$
3. Só aceitamos a face se ela aparecer pelo menos **K** vezes no histórico recente:
   $$
   count(id) \ge K
   $$

---

# Passo B — Detecção de Atividades (Pose Estimation)

Utiliza **MediaPipe Pose** para extrair 33 *landmarks* corporais. A classificação é feita via **Heurísticas Geométricas** (sem necessidade de treino de nova rede), o que mantém o sistema leve.

As coordenadas são normalizadas $P(x, y)$, onde $y$ cresce de cima para baixo.

## Heurística 1: Braços Levantados (Anomalia)

Consideramos "Braços Levantados" quando os punhos ($Wrist$) estão acima ($y$ menor) do nariz ($Nose$).

$$
y_{LeftWrist} < y_{Nose} \quad \text{E} \quad y_{RightWrist} < y_{Nose}
$$

## Heurística 2: Mão no Rosto (Pensativo/Espanto)

Calculamos a **Distância Euclidiana** $d$ entre o punho e o nariz. Se for menor que um limiar $\epsilon$ (0.15), classifica-se como mão no rosto.

$$
d(P_1, P_2) = \sqrt{(x_2 - x_1)^2 + (y_2 - y_1)^2}
$$

A condição é: $d(Wrist, Nose) < 0.15$.

## Heurística 3: Braços Cruzados (Defensivo)

Verificamos se os punhos estão na altura do tórax e se a distância horizontal entre eles é mínima (cruzamento):

$$
|x_{LeftWrist} - x_{RightWrist}| < 0.15
$$

---

# Passo C — Consolidação

O resumo final processa os logs gerados (`stepA_summary.txt` e `stepB_summary.txt`) para gerar estatísticas consolidadas.

## Definição de Anomalia

A métrica de anomalia $A$ é definida pela soma de ocorrências da classe "Braços Levantados" (movimento brusco/atípico):

$$
A_{total} = \sum_{i=0}^{N} [Class(frame_i) == Activity_{raised}]
$$