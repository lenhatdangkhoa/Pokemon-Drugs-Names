import pandas as pd
import matplotlib.pyplot as plt

csv_path = "evaluation/table_brand_gpt4o_rag/confabulation_rates_table.csv"
df = pd.read_csv(csv_path)

# Matplotlib table figure
fig, ax = plt.subplots(figsize=(20, 10))  # wider and taller
ax.axis("off")

table = ax.table(
    cellText=df.values,
    colLabels=df.columns,
    loc="center",
    cellLoc="left",
)

table.auto_set_font_size(False)
table.set_fontsize(9)      # slightly larger text
table.scale(1.5, 2.0)      # widen and heighten cells

plt.title(
    "Confabulation rates – GPT-4o-mini with online RAG (brand & generic)",
    pad=20,
)

plt.tight_layout()
plt.savefig("confabulation_table_rag_matplotlib.png", dpi=300)
plt.close(fig)