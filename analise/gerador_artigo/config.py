# --- SINCRONIZAÇÃO COM INEP ---
# Os anos simulados são os anos presentes no CSV consolidado do INEP (anos de ingresso com âncora real).
# ANOS_SEM_INEP só é usado no modo sem INEP (ver mock_data_generator.py), e as turmas de ingresso
# geradas nesse modo são marcadas como não ancoradas. Equivale aos anos coletados pelo pipeline do INEP.
ANOS_SEM_INEP = list(range(2010, 2025))

# --- REPRODUTIBILIDADE ---
SEMENTE = 42

# --- PARÂMETROS DE VOLUME ---
# Usado apenas no modo sem INEP (desenvolvimento), nunca para completar anos sem âncora.
NUM_ALUNOS_POR_TURMA = 50

# --- ESTATÍSTICAS SOCIOECONÔMICAS (Referencial Leonel Jr. & Melo) ---
P_TRABALHA = 0.62           # Porcentagem de alunos que trabalham
P_DESLOCAMENTO = 0.35       # Porcentagem de alunos que moram em outra cidade
P_VULNERAVEL = 0.40         # Índice de vulnerabilidade social

# --- DINÂMICA ACADÊMICA ---
# Nível Spaghetti (0.0 a 1.0): probabilidade, por semestre, de o aluno adiantar uma disciplina
# de período futuro no lugar de uma do período corrente. A disciplina preterida fica pendente
# e é cursada depois, de modo que a matriz continua tendo de ser integralizada.
NIVEL_SPAGHETTI = 0.25

CARGA_MAXIMA = 7            # Disciplinas por semestre (pendências + período corrente)
PRAZO_MAXIMO = 16           # Semestres cursados até o desligamento por prazo (2x a duração)
P_TRANCAMENTO = 0.06        # Probabilidade base de trancar um semestre
P_TRANCAMENTO_TRABALHA = 0.04  # Acréscimo para quem trabalha
P_EVASAO_BASE = 0.02        # Risco de evasão por semestre, sem reprovações
P_EVASAO_POR_REPROVA = 0.05 # Acréscimo por reprovação no semestre
P_EVASAO_MAX = 0.80

# --- CALIBRAÇÃO DE DESFECHO (INEP) ---
# Indicador: concluintes do ano t / ingressantes do ano t - DEFASAGEM, somados na janela de anos
# (razão agregada, estável em volumes pequenos). O mesmo indicador é calculado no dado sintético.
DEFASAGEM_CONCLUSAO = 3     # Anos entre o ingresso e a conclusão no prazo (8 semestres: ingresso em t-3, conclusão em t)
ANOS_AQUECIMENTO = 2        # Anos iniciais descartados: o sintético não tem ingressantes anteriores a 2010
JANELA_MOVEL = 3            # Anos por janela móvel na aba de validação

# --- MATRIZ CURRICULAR (Sistemas de Informação - UniCatólica) ---
MATRIZ = {
    1: ["Arquitetura de Computadores", "Estudos Socioantropológicos", "Fundamentos de Sistemas de Informação", "Gestão de Tecnologia da Informação", "Raciocínio Lógico e Estatística"],
    2: ["Fundamentos Teológicos", "Inglês Técnico", "Lógica de Programação", "Matemática", "Sistemas Operacionais"],
    3: ["Design de Sistemas de Informação", "Fundamentos de Banco de Dados", "Programação Orientada a Objetos", "Redes de Computadores I", "Tópicos Teológicos"],
    4: ["Análise e Projeto de Sistemas I", "Estruturas de Dados", "Legislação Empresarial e Trabalhista", "Psicologia Organizacional", "Redes de Computadores II"],
    5: ["Engenharia de Software", "Programação para Web", "Segurança e Auditoria", "Técnicas de BD", "Estágio I"],
    6: ["Big Data", "Gestão de Projetos", "Linguagens Web", "Metodologia Científica", "Sistemas Distribuídos", "Estágio II"],
    7: ["Computação e Algoritmos", "Inteligência Artificial", "Prog. Aplicativos Móveis", "Projeto Final: Implementação", "Responsabilidade Social", "Estágio III"],
    8: ["Análise e Projeto de Sistemas II", "Gestão de Carreira", "Projeto Final: Documentação", "Supply Chain", "Tópicos Especiais", "Estágio IV"]
}
PERIODO = {d: p for p, ds in MATRIZ.items() for d in ds}

# Pré-requisitos em cadeia: não podem ser adiantados nem cursados antes do anterior aprovado
PRE_REQUISITOS = {
    "Estágio II": "Estágio I", "Estágio III": "Estágio II", "Estágio IV": "Estágio III",
    "Projeto Final: Documentação": "Projeto Final: Implementação",
}

# --- DISCIPLINAS GARGALO (Motores de Evasão - Garcia, 2020) ---
GARGALOS = ["Lógica de Programação", "Matemática", "Estruturas de Dados", "Computação e Algoritmos", "Linguagens Web"]
