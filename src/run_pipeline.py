import sys
import os

# Adiciona o diretório atual ao path para encontrar os módulos
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from step_a_faces_emotions import run_faces_emotions
from step_b_activities import run_activities
# Se você salvou o arquivo anterior como step_c_summary.py:
from step_c_summary import gerar_relatorio 

def main():
    print("🚀 INICIANDO PIPELINE COMPLETO - TECH CHALLENGE 4")
    
    # 1. Executa Step A (Detecta Rostos, Emoções e Identidade)
    # Gera: outputs/stepA_annotated.mp4 e outputs/stepA_summary.txt
    run_faces_emotions()
    
    # 2. Executa Step B (Detecta Atividades no corpo)
    # Lê o vídeo do Step A e gera: outputs/stepB_final.mp4 e outputs/stepB_summary.txt
    run_activities()
    
    # 3. Executa Step C (Gera Relatório Final)
    # Lê os txts anteriores e gera: outputs/relatorio_final.txt
    gerar_relatorio()

    print("\n🏁 PIPELINE FINALIZADO!")
    print("Verifique a pasta 'outputs' para os resultados.")

if __name__ == "__main__":
    main()