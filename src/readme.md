# Passo A — Reconhecimento facial + Emoções (com robustez)

Este passo implementa:

- **Reconhecimento facial (detecção e marcação):** localizar rostos no vídeo e desenhar *bounding boxes*.
- **Análise de expressões emocionais:** para cada rosto detectado, inferir a **emoção dominante** e exibir a etiqueta no vídeo.

Além do “básico”, o Passo A inclui três melhorias (“patches”) para tornar o pipeline mais **genérico**, **robusto** e menos dependente de **ajustes manuais**:

1. **Autoajuste estatístico** (warm-up + percentis)  
2. **Filtro por confiança do detector**  
3. **Persistência temporal** (consistência no tempo)

---

## Visão geral do pipeline

O vídeo é lido frame a frame, mas a análise “pesada” é feita por **amostragem temporal**:

- Definimos `FRAME_STEP = N`
- Processamos apenas frames onde:

$$
frame\_idx \bmod N = 0
$$

Isso reduz custo computacional **sem perder representatividade**, pois frames adjacentes tendem a ser muito parecidos.

Em cada frame analisado:

1. Detectamos faces com `DeepFace.extract_faces`, obtendo:
   - *bounding box* `(x, y, w, h)`
   - recorte do rosto (`face_crop`)
   - `confidence` do detector
2. Aplicamos filtros (patches) para reduzir falsos positivos.
3. Estimamos emoção em cada `face_crop` com `DeepFace.analyze(actions=["emotion"])`.
4. Desenhamos *bounding boxes* e emoção no vídeo e agregamos estatísticas.

---

## Patch 1 — Autoajuste estatístico por warm-up (percentis)

### Problema

Se definirmos limiares fixos como:

- “área mínima do rosto”
- “área máxima do rosto”
- “proporção largura/altura (*aspect ratio*)”

…o sistema pode funcionar em um vídeo e falhar em outro, pois:

- a câmera pode estar mais longe/perto  
- a resolução muda  
- o enquadramento muda  

### Ideia

Em vez de escolher limiares fixos “no chute”, usamos um **warm-up**:  
coletamos amostras reais do próprio vídeo nos primeiros frames analisados e calculamos limites por **estatística robusta**.

### O que coletamos

Para cada face detectada no warm-up, coletamos:

**Área da bounding box:**

$$
A = w \times h
$$

**Aspect Ratio (AR):**

$$
AR = \frac{w}{h}
$$

**Confiança do detector** (usada no Patch 2)

### Por que área e AR?

- Falsos positivos comuns (mão, telefone, objetos) tendem a ter áreas “fora do padrão” e **AR** muito “achatado” (muito largo/baixo) ou “estreito” (muito fino/alto).
- Rostos reais costumam cair em faixas mais estáveis de área e proporção, dadas as condições do vídeo.

### Como definimos os limiares (percentis)

Depois de coletar um número suficiente de amostras, definimos:

**Área mínima como percentil 10:**

$$
A_{min} = P_{10}(A)
$$

**Área máxima como percentil 95:**

$$
A_{max} = P_{95}(A)
$$

**AR mínimo como percentil 5:**

$$
AR_{min} = P_{5}(AR)
$$

**AR máximo como percentil 95:**

$$
AR_{max} = P_{95}(AR)
$$

### Por que percentis (e não média)?

Percentis são **robustos a outliers**.

Exemplo:

- Se um falso positivo “tela inteira” aparecer, ele é um valor extremo.
- A média seria puxada para cima (ruim).
- O percentil 95 tende a manter um limite seguro sem “inflar” por um outlier.

### Como o filtro funciona (após warm-up)

Uma detecção só é aceita se:

$$
A_{min} \le A \le A_{max}
$$

e

$$
AR_{min} \le AR \le AR_{max}
$$

Isso torna o pipeline **autoajustável** a diferentes vídeos.

---

## Patch 2 — Filtro por confiança do detector

### Problema

Mesmo com área e proporção plausíveis, o detector pode estar “na dúvida” e retornar falsas faces.

O `DeepFace.extract_faces` retorna um valor de confiança:

- \( confidence \in [0,1] \) (em geral)
- Quanto maior, mais convicto o detector está de que a região é uma face.

### Ideia

Descartar detecções com confiança baixa.

### Como definimos o limiar automaticamente

Durante o warm-up, também coletamos `confidence` e definimos:

$$
conf_{min} = P_{20}(confidence)
$$

Ou seja: mantemos somente detecções acima de um limiar que reflete o “padrão do vídeo”, sem fixar um número absoluto.

### Como o filtro funciona

Uma face só é aceita se:

$$
confidence \ge conf_{min}
$$

Isso reduz falsos positivos como:

- objetos com padrões parecidos com rosto
- rostos parcialmente ocluídos em condições ruins (dependendo do objetivo)

---

## Patch 3 — Persistência temporal (consistência ao longo do tempo)

### Problema

Muitos falsos positivos aparecem por 1 frame e somem no seguinte.

Exemplos:

- um objeto com sombra/ângulo específico em um frame único
- um reflexo

### Ideia

Exigir que uma detecção seja consistente por pelo menos **K** frames analisados.

### Como fazemos na prática

Criamos um “id aproximado” da face baseado na posição da bounding box.

Exemplo (quantização em grade):

$$
id = \left(\mathrm{round}\left(\frac{x}{20}\right), \ \mathrm{round}\left(\frac{y}{20}\right)\right)
$$

- Isso agrupa pequenas variações de posição como sendo a “mesma” face aproximada.
- Guardamos os últimos IDs em um histórico.

### Regra de persistência

Só aceitamos a face se ela aparecer pelo menos **K** vezes no histórico:

$$
count(id) \ge K
$$

Com \(K=2\), por exemplo:

- detecção que aparece em 1 frame e some → **é descartada**
- rosto real (que persiste) → **é mantido**

Isso melhora a estabilidade visual e reduz “caixas piscando”.

---

## Agregação temporal das emoções (resumo parcial do Passo A)

Para cada frame analisado e cada face válida, obtemos a emoção dominante \(e\) e incrementamos um contador:

$$
count(e) \leftarrow count(e) + 1
$$

Isso cria um “perfil emocional” ao longo do vídeo.

### Por que agregação é importante?

- Emoção em um frame isolado pode ser ruído.
- A agregação temporal reduz a influência de instantes pontuais.

## Agregação temporal das emoções (resumo parcial do Passo A)

Para cada frame analisado e cada face válida, obtemos a emoção dominante \(e\) e incrementamos um contador:

$$
count(e) \leftarrow count(e) + 1
$$

Isso cria um “perfil emocional” ao longo do vídeo.

## Observação sobre áudio

O OpenCV gera o vídeo anotado **sem trilha de áudio** (o `VideoWriter` não faz mux de áudio).  
Isso não impacta a funcionalidade do desafio, pois a evidência visual (caixas e labels) é o foco do Passo A.

---

## Resultado do Passo A

O Passo A gera:

- `outputs/stepA_annotated.mp4`: vídeo anotado com bounding boxes + emoção por face
- `outputs/stepA_summary.txt`: resumo com:
  - frames totais e frames analisados
  - total de faces detectadas
  - top emoções
