import pandas as pd

#-------------------------------------------------------------------- the path to the results----------------------------------------------------------------------------------------#

#path = "D:/APHRC/ABM/fpsim/results.csv"

#results_data = pd.read_csv(path)

#print(results_data.head())

#------------------------------------------------------------creating the age grpups and contraceptive use----------------------------------------------------------------------------------#



import plotly.graph_objects as go

# Family planning method labels and colors
labels = ["None", "Withdrawal", "Other/trad.", "Condom", "Pill", "Injectable", "Implant", "IUD", "Female sterilization", "Other/mod."]
colors = ["blue", "lightblue", "orange", "lightsalmon", "green", "lightgreen", "red", "lightcoral", "purple", "mediumpurple"]

# Switching matrices (example data for one age group)
switching_matrices = {
    '<25': [
        [0.5, 0.05, 0.05, 0.1, 0.1, 0.05, 0.05, 0.05, 0.025, 0.025],
        [0.1, 0.6, 0.05, 0.05, 0.05, 0.05, 0.025, 0.025, 0.015, 0.015],
        [0.1, 0.05, 0.6, 0.05, 0.05, 0.05, 0.025, 0.025, 0.015, 0.015],
        [0.1, 0.05, 0.05, 0.5, 0.1, 0.1, 0.05, 0.05, 0.025, 0.025],
        [0.05, 0.05, 0.05, 0.1, 0.5, 0.1, 0.05, 0.05, 0.025, 0.025],
        [0.05, 0.05, 0.05, 0.1, 0.1, 0.5, 0.05, 0.05, 0.025, 0.025],
        [0.05, 0.05, 0.05, 0.1, 0.1, 0.1, 0.5, 0.05, 0.025, 0.025],
        [0.05, 0.05, 0.05, 0.1, 0.1, 0.1, 0.05, 0.5, 0.025, 0.025],
        [0.05, 0.05, 0.05, 0.1, 0.1, 0.1, 0.05, 0.05, 0.5, 0.025],
        [0.05, 0.05, 0.05, 0.1, 0.1, 0.1, 0.05, 0.05, 0.025, 0.5]
    ]
}

# Prepare Sankey diagram data
def prepare_sankey_data(matrix, colors):
    sources, targets, values, link_colors = [], [], [], []
    for i in range(len(matrix)):
        for j in range(len(matrix[i])):
            sources.append(i)
            targets.append(j + len(matrix))  # Offset target indices
            values.append(matrix[i][j])
            link_colors.append(colors[i])
    return sources, targets, values, link_colors

# Prepare data for '<25' age group as an example
age_group = '<25'
sources, targets, values, link_colors = prepare_sankey_data(switching_matrices[age_group], colors)

# Create the Sankey diagram
fig = go.Figure(data=[go.Sankey(
    node=dict(
        pad=15,
        thickness=20,
        line=dict(color="black", width=0.5),
        label=labels + labels,  # Duplicate labels for source and target
        color=colors * 2  # Duplicate colors for source and target
    ),
    link=dict(
        source=sources,
        target=targets,
        value=values,
        color=link_colors
    )
)])

# Update layout
fig.update_layout(title_text=f'Switching Matrix for Age Group {age_group}', font_size=10)

# Show the plot
fig.show()
