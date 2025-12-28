import os
import re

# ============================================================
# CONFIGURAÇÕES
# ============================================================
ARQUIVO_A = "outputs/stepA_summary.txt"
ARQUIVO_B = "outputs/stepB_summary.txt"
SAIDA_FINAL = "outputs/relatorio_final.txt"

def ler_arquivo(caminho):
    """Lê o conteúdo de um arquivo se ele existir."""
    if not os.path.exists(caminho):
        print(f"⚠️ Aviso: Arquivo não encontrado: {caminho}")
        return ""
    with open(caminho, "r", encoding="utf-8") as f:
        return f.read()

def extrair_dados_a(texto):
    """
    Busca linhas como '- happy: 45' usando Expressões Regulares (Regex).
    Retorna um dicionário: {'happy': 45, 'neutral': 10...}
    """
    # Procura padrão: hífen, palavra(emoção), dois pontos, número
    padrao = r"- (\w+): (\d+)"
    matches = re.findall(padrao, texto)
    return {emocao: int(qtd) for emocao, qtd in matches}

def extrair_dados_b(texto):
    """
    Busca linhas como '- Bracos Levantados: 12 frames'.
    """
    # Procura padrão: hífen, texto livre(atividade), dois pontos, número
    padrao = r"- ([\w\s/]+): (\d+) frames"
    matches = re.findall(padrao, texto)
    return {atv.strip(): int(qtd) for atv, qtd in matches}

def gerar_relatorio():
    print("⏳ Iniciando Step C: Geração de Relatório...")

    # 1. Leitura dos dados brutos
    txt_a = ler_arquivo(ARQUIVO_A)
    txt_b = ler_arquivo(ARQUIVO_B)

    if not txt_a or not txt_b:
        print("❌ Erro: Faltam arquivos de resumo dos passos anteriores.")
        return

    # 2. Processamento (Parsing)
    emocoes = extrair_dados_a(txt_a)
    atividades = extrair_dados_b(txt_b)

    # 3. Regra de Negócio (Definição de Anomalia)
    # Conforme o PDF: "movimento anômalo... gestos bruscos"
    # Vamos considerar "Braços Levantados" como a anomalia principal
    qtd_anomalias = atividades.get("Bracos Levantados", 0)
    
    # 4. Construção do Texto Final (Template)
    relatorio = []
    relatorio.append("========================================")
    relatorio.append("   RELATÓRIO FINAL - TECH CHALLENGE 4   ")
    relatorio.append("========================================\n")
    
    relatorio.append("1. RESUMO EXECUTIVO")
    relatorio.append("-------------------")
    relatorio.append(f"O vídeo foi processado para análise comportamental e emocional.")
    
    if qtd_anomalias > 0:
        relatorio.append(f"⚠️ ALERTA: Foram detectadas {qtd_anomalias} ocorrências de anomalia (Gestos Bruscos/Braços Levantados).")
        relatorio.append("Recomenda-se revisão humana nesses trechos.\n")
    else:
        relatorio.append("✅ Nenhuma anomalia grave detectada.\n")

    relatorio.append("2. ANÁLISE DE ATIVIDADES (STEP B)")
    relatorio.append("---------------------------------")
    for atv, qtd in atividades.items():
        relatorio.append(f"- {atv}: {qtd} frames")
    relatorio.append("")

    relatorio.append("3. ANÁLISE EMOCIONAL (STEP A)")
    relatorio.append("-----------------------------")
    relatorio.append("Distribuição das emoções detectadas nos rostos:")
    # Ordenar emoções da mais frequente para a menos frequente
    emocoes_ordenadas = sorted(emocoes.items(), key=lambda x: x[1], reverse=True)
    for emo, qtd in emocoes_ordenadas:
        relatorio.append(f"- {emo}: {qtd}")
    
    relatorio.append("\n========================================")
    relatorio.append("Gerado automaticamente pelo Pipeline IADT.")

    # 5. Salvar em arquivo
    texto_completo = "\n".join(relatorio)
    with open(SAIDA_FINAL, "w", encoding="utf-8") as f:
        f.write(texto_completo)

    print(f"✅ Relatório gerado com sucesso: {SAIDA_FINAL}")
    print("-" * 30)
    print(texto_completo) # Mostra no terminal também
    print("-" * 30)

if __name__ == "__main__":
    gerar_relatorio()