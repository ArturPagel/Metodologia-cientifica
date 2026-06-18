"""
================================================================
 CRISPR-Cas9 — Edição Genética na Fecundação
 Doença  : Huntington (gene HTT)
 Rede    : Homo sapiens PPI — STRING v12.0 (9606)
 Modelo  : Embrião editado vs não-editado na rede real
================================================================

Como rodar
----------
  venv\\Scripts\\activate
  pip install networkx matplotlib pandas numpy scipy
  python crispr_htt_ppi.py

Coloque o arquivo STRING na mesma pasta com QUALQUER um desses nomes:
  bla / bla.gz / bla.txt / bla.txt.gz
  9606.protein.links.full.v12.0.txt.gz
O script detecta automaticamente se é gzip ou texto puro.
================================================================
"""

import os, gzip, random
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from matplotlib.animation import FuncAnimation

# ================================================================
# PARÂMETROS — ajuste aqui se precisar
# ================================================================

CANDIDATOS_ARQUIVO = [
    "bla", "bla.gz", "bla.txt", "bla.txt.gz",
    "9606.protein.links.full.v12.0.txt.gz",
    "9606.protein.links.full.v12.0.txt",
]

PROTEINA_HTT      = "9606.ENSP00000347184"  # HTT humano no STRING
SCORE_MINIMO      = 700     # interações de alta confiança (0–1000)
MAX_NOS           = 300     # tamanho do subgrafo ao redor do HTT
EFICIENCIA_CRISPR = 0.92    # 92% de sucesso na edição do zigoto
TAXA_OFF_TARGET   = 0.02    # 2% de corte fora do alvo por passo
PROB_DANO         = 0.18    # probabilidade de dano por passo (HTT+)
PROB_CORRECAO     = 0.10    # probabilidade de restauração (editado)
PASSOS_SIM        = 50      # passos de propagação
PAUSA_MS          = 200     # milissegundos entre frames da animação
SEED              = 42

random.seed(SEED)
np.random.seed(SEED)

# ================================================================
# 1. LOCALIZAR E CARREGAR O ARQUIVO
# ================================================================

def eh_gzip(caminho: str) -> bool:
    """Verifica os 2 bytes mágicos do gzip."""
    with open(caminho, "rb") as f:
        return f.read(2) == b"\x1f\x8b"

def carregar_dados(caminho: str) -> pd.DataFrame:
    compressao = "gzip" if eh_gzip(caminho) else None
    tipo = "gzip" if compressao else "texto puro"
    print(f"  Formato detectado : {tipo}")
    df = pd.read_csv(
        caminho,
        sep=r"\s+",
        compression=compressao,
        usecols=["protein1", "protein2", "combined_score"],
        dtype={"protein1": str, "protein2": str, "combined_score": int},
    )
    return df

print("=" * 60)
print("  CRISPR-Cas9 — HTT / Huntington — STRING Homo sapiens")
print("=" * 60)

arquivo = None
for nome in CANDIDATOS_ARQUIVO:
    if os.path.exists(nome):
        arquivo = nome
        break

if arquivo is None:
    print("\nERRO: nenhum arquivo de dados encontrado.")
    print("Arquivos procurados:")
    for n in CANDIDATOS_ARQUIVO:
        print(f"  {n}")
    print("\nBaixe em:")
    print("  https://stringdb-downloads.org/download/")
    print("  protein.links.full.v12.0/9606.protein.links.full.v12.0.txt.gz")
    raise SystemExit(1)

print(f"\n  Arquivo encontrado : {arquivo}")
dados = carregar_dados(arquivo)
print(f"  Interações totais  : {len(dados):,}")

dados = dados[dados["combined_score"] >= SCORE_MINIMO].reset_index(drop=True)
print(f"  Após filtro ≥{SCORE_MINIMO}   : {len(dados):,}")

# ================================================================
# 2. LOCALIZAR HTT NA REDE
# ================================================================

proteinas = set(dados["protein1"]) | set(dados["protein2"])

if PROTEINA_HTT not in proteinas:
    # Busca parcial pelo ID Ensembl
    candidatos = [p for p in proteinas if "ENSP00000347184" in p]
    if candidatos:
        PROTEINA_HTT = candidatos[0]
        print(f"\n  HTT encontrado    : {PROTEINA_HTT}")
    else:
        # Usa o hub de maior grau como substituto
        G_tmp = nx.from_pandas_edgelist(dados.head(50_000), "protein1", "protein2")
        PROTEINA_HTT = max(G_tmp.nodes(), key=lambda n: G_tmp.degree(n))
        print(f"\n  AVISO: HTT não encontrado. Usando hub: {PROTEINA_HTT}")
else:
    print(f"\n  HTT localizado    : {PROTEINA_HTT}")

# ================================================================
# 3. CONSTRUIR SUBGRAFO (BFS a partir do HTT)
# ================================================================

print(f"\n  Construindo subgrafo (máx {MAX_NOS} nós)...")

G_full = nx.from_pandas_edgelist(
    dados, "protein1", "protein2", edge_attr="combined_score"
)

nos = {PROTEINA_HTT}
fila = [PROTEINA_HTT]

while fila and len(nos) < MAX_NOS:
    atual = fila.pop(0)
    if atual not in G_full:
        continue
    vizinhos = sorted(
        G_full.neighbors(atual),
        key=lambda v: G_full[atual][v].get("combined_score", 0),
        reverse=True,
    )
    for v in vizinhos:
        if len(nos) >= MAX_NOS:
            break
        if v not in nos:
            nos.add(v)
            fila.append(v)

G = G_full.subgraph(nos).copy()
del G_full  # liberar memória

print(f"  Nós              : {G.number_of_nodes()}")
print(f"  Arestas          : {G.number_of_edges()}")

# ================================================================
# 4. MÉTRICAS DA REDE
# ================================================================

graus       = dict(G.degree())
grau_medio  = np.mean(list(graus.values()))
hubs_top5   = sorted(graus, key=graus.get, reverse=True)[:5]
n_comp      = nx.number_connected_components(G)

print(f"\n  Grau médio        : {grau_medio:.2f}")
print(f"  Componentes       : {n_comp}")
print(f"  Grau do HTT       : {graus.get(PROTEINA_HTT, 0)}")
print("  Top 5 hubs:")
for h in hubs_top5:
    print(f"    {h}  grau={graus[h]}")

# ================================================================
# 5. LAYOUT
# ================================================================

print("\n  Calculando layout spring...")
pos = nx.spring_layout(G, seed=SEED, k=0.25, iterations=80)
print("  Layout concluído.")

# ================================================================
# 6. MODELO DE ESTADOS
# ================================================================
#
#  "normal"     — proteína saudável
#  "htt_pos"    — carrega alelo HTT mutante (dominante)
#  "danificado" — atingido em cascata pelo HTT+
#  "editado"    — corrigido pelo CRISPR-Cas9
#  "off_target" — corte fora do alvo pelo Cas9

ESTADOS = ["normal", "htt_pos", "danificado", "editado", "off_target"]

CORES = {
    "normal"    : "#27ae60",
    "htt_pos"   : "#e74c3c",
    "danificado": "#8e44ad",
    "editado"   : "#2980b9",
    "off_target": "#f39c12",
}

LABELS = {
    "normal"    : "Normal",
    "htt_pos"   : "HTT+ (mutante)",
    "danificado": "Danificado (cascata)",
    "editado"   : "Editado CRISPR",
    "off_target": "Off-target",
}

def init_estados(com_crispr: bool) -> dict:
    est = {n: "normal" for n in G.nodes()}
    if com_crispr:
        est[PROTEINA_HTT] = "editado" if random.random() < EFICIENCIA_CRISPR else "htt_pos"
    else:
        est[PROTEINA_HTT] = "htt_pos"
    return est

def propagar(est: dict, com_crispr: bool) -> dict:
    novo = est.copy()
    for no in G.nodes():
        e = est[no]

        # HTT+ danifica vizinhos normais (proporcional ao score)
        if e == "htt_pos":
            for viz in G.neighbors(no):
                if est[viz] == "normal":
                    score  = G[no][viz].get("combined_score", 500)
                    chance = (score / 1000) * PROB_DANO
                    if random.random() < chance:
                        novo[viz] = "danificado"

        # Editado restaura vizinhos danificados / HTT+
        elif e == "editado":
            for viz in G.neighbors(no):
                if est[viz] in ("danificado", "htt_pos"):
                    score  = G[no][viz].get("combined_score", 500)
                    chance = (score / 1000) * PROB_CORRECAO
                    if random.random() < chance:
                        novo[viz] = "editado"

        # Off-target aleatório (raro)
        if com_crispr and e == "normal":
            if random.random() < TAXA_OFF_TARGET * 0.08:
                novo[no] = "off_target"

    return novo

# ================================================================
# 7. SIMULAÇÕES
# ================================================================

print("\n  Simulando embrião SEM edição...")
hist_d = [init_estados(False)]
for _ in range(PASSOS_SIM):
    hist_d.append(propagar(hist_d[-1], False))

print("  Simulando embrião COM CRISPR-Cas9...")
hist_e = [init_estados(True)]
for _ in range(PASSOS_SIM):
    hist_e.append(propagar(hist_e[-1], True))

# ================================================================
# 8. SÉRIES TEMPORAIS
# ================================================================

def series(historico):
    s = {e: [] for e in ESTADOS}
    total = G.number_of_nodes()
    for snap in historico:
        cnt = {e: 0 for e in ESTADOS}
        for v in snap.values():
            cnt[v] += 1
        for e in ESTADOS:
            s[e].append(cnt[e] / total * 100)
    return s

ser_d = series(hist_d)
ser_e = series(hist_e)
tt    = list(range(PASSOS_SIM + 1))

# ================================================================
# 9. FIGURA
# ================================================================

plt.style.use("dark_background")
BG   = "#080e1a"
AX_BG = "#0d1626"

fig = plt.figure(figsize=(20, 10), facecolor=BG)
fig.suptitle(
    "CRISPR-Cas9  ·  Edição na Fecundação  ·  Huntington (HTT)  ·  Rede PPI Humana — STRING v12.0",
    fontsize=14, color="#ddeeff", fontweight="bold", y=0.977,
)

gs = gridspec.GridSpec(
    2, 3, figure=fig,
    hspace=0.46, wspace=0.34,
    left=0.05, right=0.97,
    top=0.93, bottom=0.09,
)

ax_rd = fig.add_subplot(gs[0, 0])   # rede sem CRISPR
ax_re = fig.add_subplot(gs[0, 1])   # rede com CRISPR
ax_nf = fig.add_subplot(gs[0, 2])   # painel info
ax_ld = fig.add_subplot(gs[1, 0])   # linha sem CRISPR
ax_le = fig.add_subplot(gs[1, 1])   # linha com CRISPR
ax_cmp= fig.add_subplot(gs[1, 2])   # comparação barras

for ax in [ax_rd, ax_re, ax_nf, ax_ld, ax_le, ax_cmp]:
    ax.set_facecolor(AX_BG)
    for sp in ax.spines.values():
        sp.set_edgecolor("#1a2d45")

# Legenda global
patches = [mpatches.Patch(color=CORES[e], label=LABELS[e]) for e in ESTADOS]
fig.legend(
    handles=patches, loc="lower center", ncol=5,
    facecolor="#0d1a2b", labelcolor="white",
    fontsize=8.5, framealpha=0.9,
    bbox_to_anchor=(0.5, 0.005),
)

# Tamanho dos nós (HTT maior)
node_sz = [
    200 if n == PROTEINA_HTT else max(12, 15 + graus.get(n, 1) * 0.3)
    for n in G.nodes()
]

# ── Funções de desenho ──────────────────────────────────────────

def _style_ax(ax):
    ax.set_facecolor(AX_BG)
    for sp in ax.spines.values():
        sp.set_edgecolor("#1a2d45")

def draw_rede(ax, snap, titulo, passo):
    ax.clear(); _style_ax(ax)
    cores = [CORES[snap[n]] for n in G.nodes()]
    nx.draw_networkx_edges(G, pos, ax=ax, alpha=0.07,
                           width=0.3, edge_color="white")
    nx.draw_networkx_nodes(G, pos, ax=ax,
                           node_color=cores, node_size=node_sz,
                           linewidths=0.3, edgecolors="#ffffff")
    # Destaque HTT
    nx.draw_networkx_nodes(G, pos, ax=ax,
                           nodelist=[PROTEINA_HTT],
                           node_color=[CORES[snap[PROTEINA_HTT]]],
                           node_size=[320], linewidths=2.0,
                           edgecolors="#ffffff")
    ax.set_title(f"{titulo}  ·  t = {passo}",
                 color="#aaccee", fontsize=10, pad=6)
    ax.axis("off")

def draw_linha(ax, ser, titulo, passo):
    ax.clear(); _style_ax(ax)
    x = tt[:passo + 1]
    for e in ESTADOS:
        ax.plot(x, ser[e][:passo + 1], color=CORES[e],
                linewidth=2.2, label=LABELS[e])
    ax.set_xlim(-0.3, PASSOS_SIM + 0.3)
    ax.set_ylim(-1, 105)
    ax.set_xlabel("Passo", color="#667799", fontsize=8)
    ax.set_ylabel("% proteínas", color="#667799", fontsize=8)
    ax.set_title(titulo, color="#aaccee", fontsize=10, pad=6)
    ax.tick_params(colors="#556677", labelsize=7)

def draw_comp(ax, passo):
    ax.clear(); _style_ax(ax)
    cats  = ["htt_pos", "danificado", "editado"]
    lbls  = ["HTT+", "Danificado", "Editado"]
    x     = np.arange(len(cats))
    w     = 0.36
    vd    = [ser_d[c][passo] for c in cats]
    ve    = [ser_e[c][passo] for c in cats]
    ax.bar(x - w/2, vd, width=w, color="#e74c3c", label="Sem CRISPR", alpha=0.88)
    ax.bar(x + w/2, ve, width=w, color="#2980b9", label="Com CRISPR",  alpha=0.88)
    ax.set_xticks(x); ax.set_xticklabels(lbls, color="#aaaaaa", fontsize=8)
    ax.set_ylabel("% proteínas", color="#667799", fontsize=8)
    ax.set_ylim(0, 105)
    ax.set_title("Comparação direta", color="#aaccee", fontsize=10, pad=6)
    ax.tick_params(colors="#556677", labelsize=7)
    ax.legend(fontsize=8, facecolor="#0d1626", labelcolor="white")

def draw_info(ax, passo):
    ax.clear(); _style_ax(ax); ax.axis("off")

    afet_d  = ser_d["htt_pos"][passo] + ser_d["danificado"][passo]
    afet_e  = ser_e["htt_pos"][passo] + ser_e["danificado"][passo]
    edit_e  = ser_e["editado"][passo]
    offt_e  = ser_e["off_target"][passo]
    reducao = max(0.0, afet_d - afet_e)

    linhas = [
        ("Passo",              f"{passo} / {PASSOS_SIM}"),
        ("Proteínas na rede",  f"{G.number_of_nodes()}"),
        ("Interações",         f"{G.number_of_edges()}"),
        ("Grau do HTT",        f"{graus.get(PROTEINA_HTT, 0)}"),
        ("Score mínimo",       f"{SCORE_MINIMO}"),
        ("", ""),
        ("── SEM CRISPR ──",   ""),
        ("Proteínas afetadas", f"{afet_d:.1f}%"),
        ("", ""),
        ("── COM CRISPR ──",   ""),
        ("Editadas",           f"{edit_e:.1f}%"),
        ("Off-target",         f"{offt_e:.1f}%"),
        ("Afetadas",           f"{afet_e:.1f}%"),
        ("", ""),
        ("Redução de dano",    f"{reducao:.1f} p.p."),
        ("Eficiência CRISPR",  f"{EFICIENCIA_CRISPR*100:.0f}%"),
    ]

    y = 0.97
    for k, v in linhas:
        if k == "":
            y -= 0.03; continue
        ck = "#f0c040" if k.startswith("──") else "#6688aa"
        ax.text(0.04, y, k, color=ck, fontsize=8.5,
                transform=ax.transAxes, va="top")
        ax.text(0.97, y, v, color="#e8e8e8", fontsize=8.5,
                transform=ax.transAxes, va="top", ha="right",
                fontweight="bold")
        y -= 0.063

    ax.set_title("Resumo da Simulação", color="#aaccee",
                 fontsize=10, pad=6)

# ================================================================
# 10. ANIMAÇÃO
# ================================================================

def atualizar(frame):
    p = min(frame, PASSOS_SIM)
    draw_rede(ax_rd, hist_d[p], "Sem CRISPR", p)
    draw_rede(ax_re, hist_e[p], "Com CRISPR-Cas9", p)
    draw_info(ax_nf, p)
    draw_linha(ax_ld, ser_d, "Dinâmica — Sem CRISPR", p)
    draw_linha(ax_le, ser_e, "Dinâmica — Com CRISPR", p)
    draw_comp(ax_cmp, p)

anim = FuncAnimation(
    fig, atualizar,
    frames=PASSOS_SIM + 1,
    interval=PAUSA_MS,
    repeat=True,
    repeat_delay=2500,
)

# ================================================================
# 11. RELATÓRIO NO TERMINAL
# ================================================================

print("\n" + "=" * 60)
print(f"  RELATÓRIO — Passo final {PASSOS_SIM}")
print("=" * 60)
print(f"  Proteínas na rede    : {G.number_of_nodes()}")
print(f"  Interações           : {G.number_of_edges()}")
print(f"  Grau do HTT          : {graus.get(PROTEINA_HTT, 0)}")
print()
print("  SEM CRISPR:")
print(f"    HTT+ (mutante)     : {ser_d['htt_pos'][-1]:.1f}%")
print(f"    Danificadas        : {ser_d['danificado'][-1]:.1f}%")
afet_final_d = ser_d['htt_pos'][-1] + ser_d['danificado'][-1]
print(f"    Total afetado      : {afet_final_d:.1f}%")
print()
print("  COM CRISPR-Cas9:")
print(f"    Editadas           : {ser_e['editado'][-1]:.1f}%")
print(f"    Off-target         : {ser_e['off_target'][-1]:.1f}%")
afet_final_e = ser_e['htt_pos'][-1] + ser_e['danificado'][-1]
print(f"    Total afetado      : {afet_final_e:.1f}%")
print()
print(f"  Redução de dano      : {max(0, afet_final_d - afet_final_e):.1f} p.p.")
print(f"  Eficiência CRISPR    : {EFICIENCIA_CRISPR*100:.0f}%")
print("=" * 60)

plt.show()