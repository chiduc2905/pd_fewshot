import csv
import re

csv_path = 'wandb_export_2025-12-22T15_20_34.838+07_00.csv'

with open(csv_path, encoding='utf-8') as f:
    reader = csv.DictReader(f)
    rows = list(reader)

# Parse data into structured format
data = {}

for r in rows:
    name = r.get('Name', '')
    acc = float(r.get('test_accuracy', 0)) * 100
    f1 = float(r.get('test_f1', 0)) * 100
    rec = float(r.get('test_recall', 0)) * 100
    prec = float(r.get('test_precision', 0)) * 100 if r.get('test_precision') else acc  # Use acc as proxy if no precision
    
    match = re.match(r'(\w+)_(\d+)shot_contrastive_lambda[\d.]+_(\w+)', name)
    if match:
        model = match.group(1)
        shot = match.group(2)
        samples = match.group(3)
        
        if model not in data:
            data[model] = {}
        if samples not in data[model]:
            data[model][samples] = {}
        
        data[model][samples][shot] = {'acc': acc, 'f1': f1, 'recall': rec, 'prec': prec}

models = ['protonet', 'matchingnet', 'relationnet', 'covamnet', 'dn4', 'siamese', 'cosine', 'baseline', 'feat', 'deepemd']
model_names = {
    'protonet': ('ProtoNet', 'Euclidean'),
    'matchingnet': ('MatchingNet', 'Cosine'),
    'relationnet': ('RelationNet', 'Learned'),
    'covamnet': ('CovaMNet', 'Covariance'),
    'dn4': ('DN4', 'Local Desc.'),
    'siamese': ('SiameseNet', 'Euclidean'),
    'cosine': ('Cosine', 'Cosine'),
    'baseline': ('Baseline++', 'Cosine'),
    'feat': ('FEAT', 'Transformer'),
    'deepemd': ('DeepEMD', 'EMD')
}

samples_order = ['18samples', '60samples', 'all']

# Calculate average accuracy for sorting (low to high)
model_avg = {}
for m in models:
    if m in data:
        total = 0
        count = 0
        for s in samples_order:
            if s in data[m]:
                for shot in ['1', '5']:
                    if shot in data[m][s]:
                        total += data[m][s][shot]['acc']
                        count += 1
        model_avg[m] = total / count if count > 0 else 0

# Sort models by average accuracy (low to high)
sorted_models = sorted([m for m in models if m in data], key=lambda x: model_avg.get(x, 0))

# Find max values for each column to bold
max_vals = {}
for s in samples_order:
    for shot in ['1', '5']:
        for metric in ['acc', 'prec', 'recall', 'f1']:
            key = f'{s}_{shot}_{metric}'
            max_val = 0
            for m in sorted_models:
                if m in data and s in data[m] and shot in data[m][s]:
                    val = data[m][s][shot].get(metric, 0)
                    if val > max_val:
                        max_val = val
            max_vals[key] = max_val

def fmt(val, s, shot, metric):
    key = f'{s}_{shot}_{metric}'
    if abs(val - max_vals.get(key, 0)) < 0.01:
        return f'\\textbf{{{val:.2f}}}'
    return f'{val:.2f}'

# Generate LaTeX table rows
with open('docs/results_table.tex', 'w', encoding='utf-8') as out:
    out.write('''% IEEE Transaction format table - spans full page width
% Use with \\usepackage{booktabs}, \\usepackage{multirow}, \\usepackage{array}

\\begin{table*}[!t]
\\centering
\\caption{Performance Comparison of Few-Shot Learning Methods on PD Scalogram Classification (Conv64F Backbone)}
\\label{tab:fewshot_results}
\\renewcommand{\\arraystretch}{1.1}
\\setlength{\\tabcolsep}{3pt}
\\scriptsize
\\begin{tabular}{ll|cccccc|cccccc|cccccc}
\\hline
\\multirow{3}{*}{\\textbf{Method}} & \\multirow{3}{*}{\\textbf{Metric}} & \\multicolumn{6}{c|}{\\textbf{18 samples}} & \\multicolumn{6}{c|}{\\textbf{60 samples}} & \\multicolumn{6}{c}{\\textbf{All samples}} \\\\
\\cline{3-20}
 & & \\multicolumn{2}{c}{Acc$\\uparrow$} & \\multicolumn{2}{c}{Rec$\\uparrow$} & \\multicolumn{2}{c|}{F1$\\uparrow$} & \\multicolumn{2}{c}{Acc$\\uparrow$} & \\multicolumn{2}{c}{Rec$\\uparrow$} & \\multicolumn{2}{c|}{F1$\\uparrow$} & \\multicolumn{2}{c}{Acc$\\uparrow$} & \\multicolumn{2}{c}{Rec$\\uparrow$} & \\multicolumn{2}{c}{F1$\\uparrow$} \\\\
\\cline{3-20}
 & & 1-shot & 5-shot & 1-shot & 5-shot & 1-shot & 5-shot & 1-shot & 5-shot & 1-shot & 5-shot & 1-shot & 5-shot & 1-shot & 5-shot & 1-shot & 5-shot & 1-shot & 5-shot \\\\
\\hline
''')
    
    for m in sorted_models:
        name, metric = model_names.get(m, (m, '-'))
        row = f'{name} & {metric}'
        
        for s in samples_order:
            if s in data[m]:
                d1 = data[m][s].get('1', {})
                d5 = data[m][s].get('5', {})
                
                for metric_name in ['acc', 'recall', 'f1']:
                    v1 = d1.get(metric_name, 0)
                    v5 = d5.get(metric_name, 0)
                    row += f' & {fmt(v1, s, "1", metric_name)} & {fmt(v5, s, "5", metric_name)}'
            else:
                row += ' & - & - & - & - & - & -'
        
        row += ' \\\\\n'
        out.write(row)
    
    out.write('''\\hline
\\end{tabular}
\\end{table*}
''')

print('Done - table saved to docs/results_table.tex')
