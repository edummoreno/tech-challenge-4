import os
import re

# ============================================================
# CONFIGURAÇÕES
# ============================================================
ARQUIVO_A = "outputs/stepA_summary.txt"
ARQUIVO_B = "outputs/stepB_summary.txt"
SAIDA_FINAL = "outputs/relatorio_final.txt"

EMOCOES_VALIDAS = {"angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"}


def garantir_diretorio(caminho_arquivo: str) -> None:
    """Garante que o diretório de saída existe."""
    pasta = os.path.dirname(caminho_arquivo)
    if pasta:
        os.makedirs(pasta, exist_ok=True)


def ler_arquivo(caminho: str) -> str:
    """Lê o conteúdo de um arquivo se ele existir."""
    if not os.path.exists(caminho):
        print(f"⚠️ Aviso: Arquivo não encontrado: {caminho}")
        return ""
    with open(caminho, "r", encoding="utf-8") as f:
        return f.read()


# ============================================================
# EXTRAÇÃO STEP A
# ============================================================
def extrair_meta_a(texto: str) -> dict:
    """
    Extrai metadados do Step A:
    - frames_totais
    - frames_analisados
    - frame_step
    """
    def extrair_int(padrao: str):
        m = re.search(padrao, texto)
        return int(m.group(1)) if m else None

    return {
        "frames_totais": extrair_int(r"frames_totais:\s*(\d+)"),
        "frames_analisados": extrair_int(r"frames_analisados:\s*(\d+)"),
        "frame_step": extrair_int(r"frame_step:\s*(\d+)"),
    }


def extrair_emocoes_a(texto: str) -> dict:
    """
    Extrai emoções somente do bloco:
    'Emocoes (contagem):'
    até a próxima linha vazia / próximo bloco.
    """
    linhas = texto.splitlines()
    emocoes = {}

    # Acha o início do bloco
    idx_inicio = None
    for i, linha in enumerate(linhas):
        if linha.strip().lower() == "emocoes (contagem):":
            idx_inicio = i + 1
            break

    # Se não achou o bloco, faz fallback: pega matches e filtra por emoções válidas
    if idx_inicio is None:
        matches = re.findall(r"-\s*([A-Za-z_]+):\s*(\d+)", texto)
        for k, v in matches:
            kk = k.strip()
            if kk in EMOCOES_VALIDAS:
                emocoes[kk] = int(v)
        return emocoes

    # Lê linhas do bloco até parar (linha vazia ou outro cabeçalho)
    for j in range(idx_inicio, len(linhas)):
        linha = linhas[j].strip()

        if linha == "":
            break
        # Se começar um novo bloco, para também
        if linha.lower().startswith("limiar") or linha.lower().startswith("k persistencia") or linha.lower().startswith("tamanho grid"):
            break

        m = re.match(r"-\s*([A-Za-z_]+):\s*(\d+)", linha)
        if m:
            emocao = m.group(1).strip()
            qtd = int(m.group(2))
            if emocao in EMOCOES_VALIDAS:
                emocoes[emocao] = qtd

    return emocoes


# ============================================================
# EXTRAÇÃO STEP B
# ============================================================
def extrair_meta_b(texto: str) -> dict:
    """Extrai metadados do Step B, principalmente Total Frames."""
    m = re.search(r"Total Frames:\s*(\d+)", texto)
    return {"total_frames": int(m.group(1)) if m else None}


def extrair_atividades_b(texto: str) -> dict:
    """
    Busca linhas como '- Bracos Levantados: 12 frames'.
    """
    padrao = r"-\s*([\w\s/]+):\s*(\d+)\s*frames"
    matches = re.findall(padrao, texto)
    return {atv.strip(): int(qtd) for atv, qtd in matches}


# ============================================================
# RELATÓRIO
# ============================================================
def gerar_relatorio():
    print("⏳ Iniciando Step C: Geração de Relatório...")

    # 1. Leitura dos dados brutos
    txt_a = ler_arquivo(ARQUIVO_A)
    txt_b = ler_arquivo(ARQUIVO_B)

    if not txt_a or not txt_b:
        print("❌ Erro: Faltam arquivos de resumo dos passos anteriores.")
        return

    # 2. Parsing
    meta_a = extrair_meta_a(txt_a)
    emocoes = extrair_emocoes_a(txt_a)

    meta_b = extrair_meta_b(txt_b)
    atividades = extrair_atividades_b(txt_b)

    # 3. Regra de Negócio (Anomalia)
    qtd_anomalias = atividades.get("Bracos Levantados", 0)

    # 4. Construção do Texto Final
    relatorio = []
    relatorio.append("========================================")
    relatorio.append("   RELATÓRIO FINAL - TECH CHALLENGE 4   ")
    relatorio.append("========================================\n")

    # ---- CONTEXTO DE PROCESSAMENTO (NOVO) ----
    total_video = meta_a.get("frames_totais") or meta_b.get("total_frames")
    frames_a = meta_a.get("frames_analisados")
    frame_step = meta_a.get("frame_step")
    frames_b = meta_b.get("total_frames")

    relatorio.append("📊 CONTEXTO DE PROCESSAMENTO\n")

    if total_video is not None:
        relatorio.append(f"- Total de frames do vídeo: {total_video}\n")
    else:
        relatorio.append("- Total de frames do vídeo: (não encontrado)\n")

    relatorio.append("Step A — Emoções e Faces:")
    if frames_a is not None:
        relatorio.append(f"- Frames analisados: {frames_a}")
    else:
        relatorio.append("- Frames analisados: (não encontrado)")

    if frame_step is not None:
        relatorio.append(f"- Estratégia: amostragem temporal (1 a cada {frame_step} frames)\n")
    else:
        relatorio.append("- Estratégia: amostragem temporal (frame_step não encontrado)\n")

    relatorio.append("Step B — Atividades Corporais:")
    if frames_b is not None:
        relatorio.append(f"- Frames analisados: {frames_b}")
    else:
        relatorio.append("- Frames analisados: (não encontrado)")
    relatorio.append("- Estratégia: análise frame a frame\n")

    # ---- RESUMO EXECUTIVO ----
    relatorio.append("1. RESUMO EXECUTIVO")
    relatorio.append("-------------------")
    relatorio.append("O vídeo foi processado para análise comportamental e emocional.")

    if qtd_anomalias > 0:
        relatorio.append(
            f"⚠️ ALERTA: Foram detectadas {qtd_anomalias} ocorrências de anomalia (Gestos Bruscos/Braços Levantados)."
        )
        relatorio.append("Recomenda-se revisão humana nesses trechos.\n")
    else:
        relatorio.append("✅ Nenhuma anomalia grave detectada.\n")

    # ---- STEP B ----
    relatorio.append("2. ANÁLISE DE ATIVIDADES (STEP B)")
    relatorio.append("---------------------------------")
    for atv, qtd in atividades.items():
        relatorio.append(f"- {atv}: {qtd} frames")
    relatorio.append("")

    # ---- STEP A ----
    relatorio.append("3. ANÁLISE EMOCIONAL (STEP A)")
    relatorio.append("-----------------------------")
    relatorio.append("Distribuição das emoções detectadas nos rostos:")

    if emocoes:
        emocoes_ordenadas = sorted(emocoes.items(), key=lambda x: x[1], reverse=True)
        for emo, qtd in emocoes_ordenadas:
            relatorio.append(f"- {emo}: {qtd}")
    else:
        relatorio.append("- (nenhuma emoção encontrada no resumo do Step A)")

    relatorio.append("\n========================================")
    relatorio.append("Gerado automaticamente pelo Pipeline IADT.")

    # 5. Salvar em arquivo
    garantir_diretorio(SAIDA_FINAL)
    texto_completo = "\n".join(relatorio)
    with open(SAIDA_FINAL, "w", encoding="utf-8") as f:
        f.write(texto_completo)

    print(f"✅ Relatório gerado com sucesso: {SAIDA_FINAL}")
    print("-" * 30)
    print(texto_completo)  # Mostra no terminal também
    print("-" * 30)


if __name__ == "__main__":
    gerar_relatorio()
