import matplotlib.pyplot as plt

counts = {"Mapped": 1938, "Unresolved": 1}
plt.figure(figsize=(5,4))
plt.bar(counts.keys(), counts.values(), color=["#4E79A7", "#E15759"])
plt.title("DDInter → DrugBank Mapping Success", fontsize=14, weight="bold")
plt.ylabel("Number of Entries")
for i, v in enumerate(counts.values()):
    plt.text(i, v+5, f"{v}", ha="center", fontsize=12)
plt.tight_layout()
plt.savefig("ddinter_mapping_success.png", dpi=300)
plt.show()