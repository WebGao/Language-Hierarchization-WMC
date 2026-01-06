import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.lines as mlines
import os
import re
import json
import hashlib
import stanza
from scipy.optimize import curve_fit

# -------------------------- 1. Basic Configuration --------------------------
LANGUAGES = [
    {"code": "zh-hans", "name": "Chinese", "file": "data/alice_chinese.txt", "splitter": r"[。！？；：\n\r]", "color": "#E64B35"},
    {"code": "en", "name": "English", "file": "data/alice_english.txt", "splitter": r"(?<=[.!?;:])\s+|\n\r", "color": "#4878CF"},
    {"code": "fr", "name": "French", "file": "data/alice_french.txt", "splitter": r"(?<=[.!?;:])\s+|\n\r", "color": "#13A188"},
    {"code": "de", "name": "German", "file": "data/alice_german.txt", "splitter": r"(?<=[.!?;:])\s+|\n\r", "color": "#F7DC6F"},
    {"code": "ru", "name": "Russian", "file": "data/alice_russian.txt", "splitter": r"(?<=[.!?;:])\s+|\n\r", "color": "#9370DB"},
    {"code": "ja", "name": "Japanese", "file": "data/alice_japanese.txt", "splitter": r"[。！？；：\n\r]", "color": "#2E8B57"},
    {"code": "es", "name": "Spanish", "file": "data/alice_spanish.txt", "splitter": r"(?<=[.!?;:])\s+|\n\r", "color": "#FF8C00"},
    {"code": "it", "name": "Italian", "file": "data/alice_italian.txt", "splitter": r"(?<=[.!?;:])\s+|\n\r", "color": "#8B4513"}
]

mpl.rcParams.update({
    'font.family': 'Arial', 'axes.linewidth': 1.2, 'font.size': 10,
    'axes.unicode_minus': False, 'pdf.fonttype': 42
})

STANZA_MODEL_DIR = "../stanza_models"
FIGURE_OUTPUT_DIR = "./figures"
STRUCTURE_CACHE_DIR = './structure_cache_'
WMC_MIN, WMC_MAX = 4, 9
os.makedirs(FIGURE_OUTPUT_DIR, exist_ok=True)

# -------------------------- 2. Core Computational Functions --------------------------

def calculate_open_nodes_theta(nested_sentence, count_start=1):
    word_units = {}
    def traverse_sub_component(sub_node, base_offset=0, current_depth=0):
        if isinstance(sub_node, list):
            for idx, item in enumerate(sub_node):
                if idx == 0 and isinstance(item, str):
                    word_units[item] = base_offset + current_depth + count_start
                else:
                    traverse_sub_component(item, base_offset, current_depth + 1)
        elif isinstance(sub_node, str):
            word_units[sub_node] = base_offset + current_depth + count_start
    if isinstance(nested_sentence, list):
        for sub_idx, sub_component in enumerate(nested_sentence):
            base_offset = sub_idx
            traverse_sub_component(sub_component, base_offset=base_offset)
    total_words = len(word_units)
    return sum(word_units.values()) / total_words if total_words > 0 else 0

def simulate_branch_merge(n): return list(range(n, 0, -1))
def simulate_flat_merge(n): return [n] * n

# -------------------------- 3. Plotting Functions (Nature Style) --------------------------

def plot_single_language(name, code, color, models_data):
    # Set Nature-style proportions for a compact layout
    fig, ax = plt.subplots(figsize=(4.5, 4), dpi=300)
    
    # Plot Human Working Memory Capacity (WMC) range using neutral light gray
    ax.axhspan(WMC_MIN, WMC_MAX, color='#E6E6E6', alpha=0.4, label='Human WMC', zorder=0)
    
    configs = {
        'Hierarchical': {'color': color, 'ls': '-', 'marker': 'o', 'label': 'Hierarchical'},
        'Branching':    {'color': '#555555', 'ls': '--', 'marker': 'x', 'label': 'Linear Branching'}
    }
    
    for m_name, m_cfg in configs.items():
        data = models_data[m_name]
        x, y = np.array(data['x']), np.array(data['y'])
        if len(x) < 5: continue
        
        # Scatter: Small size and low alpha to handle data overlap
        ax.scatter(x, y, s=3, alpha=0.05, color=m_cfg['color'], edgecolors='none', zorder=2)
        
        try:
            # Curve fitting
            xf = np.linspace(min(x), 65, 100)
            if m_name == 'Hierarchical':
                popt, _ = curve_fit(lambda t,a,b: a*np.log(t)+b, x, y)
                yf = popt[0]*np.log(xf) + popt[1]
            else:
                popt, _ = curve_fit(lambda t,a,b: a*t+b, x, y)
                yf = popt[0]*xf + popt[1]
            
            ax.plot(xf, yf, color=m_cfg['color'], ls=m_cfg['ls'], lw=1.5, label=m_cfg['label'], zorder=3)
        except: pass

    # Refine visual details
    ax.set_title(f'{name}', fontsize=12, fontweight='bold', loc='left')
    ax.set_xlabel('Sentence Length ($L$)', fontsize=10)
    ax.set_ylabel(r'Cognitive Load ($\theta$)', fontsize=10)
    
    # Remove top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    ax.set_xlim(2, 50)
    ax.set_ylim(0, 30)
    ax.legend(frameon=False, fontsize=8, loc='upper left')
    
    # Minimalist padding
    plt.tight_layout(pad=0.5)
    plt.savefig(os.path.join(FIGURE_OUTPUT_DIR, f"analysis_{code}.pdf"), bbox_inches='tight')
    plt.savefig(os.path.join(FIGURE_OUTPUT_DIR, f"analysis_{code}.png"), bbox_inches='tight')
    plt.close()

def plot_combined_24_lines_with_points(all_lang_data):
    # Nature-style large figure for double-column width
    fig, ax = plt.subplots(figsize=(10, 7), dpi=300)
    
    # Plot Human Working Memory Capacity (WMC) range
    ax.axhspan(WMC_MIN, WMC_MAX, color='#E6E6E6', alpha=0.4, zorder=0, label='Human WMC')
    
    xf = np.linspace(2, 50, 100) 
    model_curves = {'Hierarchical': [], 'Branching': []}
    
    # 1. Plot individual dots and fit lines for each language
    for lang in all_lang_data:
        name = lang['name']
        lang_color = lang['color']
        
        # Flag to control legend display
        label_added = False 
        
        for m_name, fit_type in [('Hierarchical', 'log'), ('Branching', 'linear')]:
            data = lang['models'][m_name]
            x, y = np.array(data['x']), np.array(data['y'])
            if len(x) < 5: continue
            
            # --- Plot raw data points ---
            # s=20, alpha=0.05 to handle high density from overlapping languages
            ax.scatter(x, y, s=20, alpha=0.05, color=lang_color, edgecolors='none', zorder=1)
            
            try:
                if fit_type == 'log':
                    popt, _ = curve_fit(lambda t,a,b: a*np.log(t)+b, x, y)
                    yf = popt[0]*np.log(xf) + popt[1]
                    ls = '-'
                else:
                    popt, _ = curve_fit(lambda t,a,b: a*t+b, x, y)
                    yf = popt[0]*xf + popt[1]
                    ls = ':' # Dot style for Branching model to differentiate
                
                model_curves[m_name].append(yf)
                
                # Legend settings
                current_label = name if not label_added else None
                ax.plot(xf, yf, color=lang_color, ls=ls, lw=1.0, alpha=0.4, label=current_label, zorder=2)
                label_added = True
            except: pass

    # 2. Plot global mean bold lines using high-contrast academic palette
    mean_cfg = {
        'Hierarchical': {'color': '#0072B2', 'ls': '-', 'label': 'MEAN: Hierarchical'},
        'Branching':    {'color': '#D55E00', 'ls': '--', 'label': 'MEAN: Branching'}
    }
    
    if model_curves['Hierarchical']:
        ax.plot(xf, np.mean(model_curves['Hierarchical'], axis=0), 
                color=mean_cfg['Hierarchical']['color'], ls=mean_cfg['Hierarchical']['ls'], 
                lw=3.0, label=mean_cfg['Hierarchical']['label'], zorder=10)
                
    if model_curves['Branching']:
        ax.plot(xf, np.mean(model_curves['Branching'], axis=0), 
                color=mean_cfg['Branching']['color'], ls=mean_cfg['Branching']['ls'], 
                lw=3.0, label=mean_cfg['Branching']['label'], zorder=10)
                
    # Visual aesthetics
    ax.set_title('Cross-linguistic Complexity Profile (Global Analysis)', fontsize=14, fontweight='bold', loc='left')
    ax.set_xlabel('Sentence Length ($L$)', fontsize=11)
    ax.set_ylabel(r'Processing Load ($\theta$)', fontsize=11)
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Coordinate range fixed to requested interval
    ax.set_xlim(2, 50)
    ax.set_ylim(0, 30)
    
    # Legend configuration: outside right
    ax.legend(frameon=False, loc='upper left', bbox_to_anchor=(1.02, 1), fontsize=8, ncol=1)
    
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURE_OUTPUT_DIR, "analysis_combined_nature.pdf"), bbox_inches='tight')
    plt.savefig(os.path.join(FIGURE_OUTPUT_DIR, "analysis_combined_nature.png"), bbox_inches='tight')
    plt.close()
# -------------------------- 4. Main Execution Logic --------------------------

if __name__ == "__main__":
    combined_results_for_plot = []
    summary_stats = []
    
    # Define dedicated cache directory for tokenization
    TOKEN_CACHE_DIR = './token_cache'
    os.makedirs(TOKEN_CACHE_DIR, exist_ok=True)
    
    print(f"{'='*20} Starting Cross-linguistic Cognitive Load Analysis {'='*20}")
    
    for lang_info in LANGUAGES:
        code, name = lang_info['code'], lang_info['name']
        if not os.path.exists(lang_info['file']):
            print(f"Warning: Corpus file not found for {name}: {lang_info['file']}")
            continue
            
        print(f"\n>>> Processing Language: {name} ({code})") 
        
        content = None
        for enc in ['utf-8', 'gbk', 'utf-8-sig', 'latin-1']:
            try:
                with open(lang_info['file'], 'r', encoding=enc) as f:
                    content = f.read()
                break
            except: continue
        if content is None: continue

        # Initialize Stanza Pipeline
        nlp = stanza.Pipeline(lang=code, processors='tokenize,pos,lemma,depparse', 
                              dir=STANZA_MODEL_DIR, logging_level='ERROR')

        raw_sents = re.split(lang_info['splitter'], content)
        temp_data = []
        
        print(f"    [Phase 1/3] Pre-tokenization and initial filtering...")
        for s in raw_sents:
            s = s.replace('\u3000', ' ').strip()
            if not s: continue
            
            if code in ["zh-hans", "ja"]:
                # --- Tokenization Cache Logic ---
                s_hash = hashlib.sha256(s.encode('utf-8')).hexdigest()
                t_cache_path = os.path.join(TOKEN_CACHE_DIR, f"{code}_{s_hash}.json")
                
                if os.path.exists(t_cache_path):
                    with open(t_cache_path, 'r', encoding='utf-8') as f:
                        words = json.load(f)
                else:
                    doc_pre = nlp(s)
                    words = [w.text for sent in doc_pre.sentences for w in sent.words if not re.match(r'^[\W\d]+$', w.text.strip())]
                    with open(t_cache_path, 'w', encoding='utf-8') as f:
                        json.dump(words, f, ensure_ascii=False)
                # -----------------------
            else:
                clean_text = re.sub(r'[^a-zA-Z0-9àáâãäåçèéêëìíîïñòóôõöùúûüýÿА-Яа-я]', ' ', s)
                words = [word for word in clean_text.split() if word]

            L = len(words)
            if L > 0: 
                clean_text = "".join(words) if code in ['zh-hans', 'ja'] else " ".join(words)
                temp_data.append({'text': clean_text, 'len': L, 'word_list': words})
        
        if not temp_data: continue

        lens = [d['len'] for d in temp_data]
        mean_len = np.mean(lens)
        std_len = np.std(lens)
        q1, q3 = np.percentile(lens, 25), np.percentile(lens, 75)
        iqr = q3 - q1
        lower, upper = max(3, q1 - 1.5*iqr), q3 + 1.5*iqr
        
        print(f"    [Statistics] Sentence Length Distribution:")
        print(f"        Total Samples: {len(lens)}")
        print(f"        Mean: {mean_len:.2f}, Std: {std_len:.2f}") 
        print(f"        Q1 (25%): {q1:.2f}, Q3 (75%): {q3:.2f}, IQR: {iqr:.2f}")
        print(f"        Outlier Filter Lower Bound: {lower:.2f}")
        print(f"        Outlier Filter Upper Bound: {upper:.2f}")
        
        filtered_entries = [d for d in temp_data if lower <= d['len'] <= upper]
        total_to_process = len(filtered_entries)
        print(f"    [Phase 2/3] Sentences remaining after filtering: {total_to_process}")

        current_lang_models = {
            'Hierarchical': {'x':[],'y':[]}, 
            'Branching': {'x':[],'y':[]}, 
            'Flat': {'x':[],'y':[]},
            'Color': lang_info['color']
        }
        
        STRUCTURE_CACHE_DIR_LANG = f"./structure_cache_{code}"
        os.makedirs(STRUCTURE_CACHE_DIR_LANG, exist_ok=True)

        print(f"    [Phase 3/3] Dependency parsing and complexity calculation:")
        for idx, entry in enumerate(filtered_entries):
            # Update progress every 5 sentences or at the end
            if (idx + 1) % 5 == 0 or (idx + 1) == total_to_process:
                print(f"\r        Progress: {idx+1}/{total_to_process} ({(idx+1)/total_to_process:.1%})", end="", flush=True)
            
            txt = entry['text']
            ckey = hashlib.sha256(f"{code}:{txt}".encode('utf-8')).hexdigest()
            cfile = os.path.join(STRUCTURE_CACHE_DIR_LANG, f"{ckey}.json")
            
            struct = None
            if os.path.exists(cfile):
                with open(cfile, 'r', encoding='utf-8') as f: struct = json.load(f)
            else:
                try:
                    doc = nlp(txt)
                    if not doc.sentences: continue
                    sent = doc.sentences[0]
                    token_dict = {w.id: {'text': w.text, 'head': w.head} for w in sent.words}
                    roots = [i for i, t in token_dict.items() if t['head'] == 0]
                    if not roots: continue
                    
                    def build_tree(idx, d):
                        token = d[idx]
                        child_ids = [k for k, t in d.items() if t['head'] == idx]
                        if not child_ids: return token['text']
                        elements = [{'p': idx, 'c': token['text']}]
                        for cid in child_ids: elements.append({'p': cid, 'c': build_tree(cid, d)})
                        elements.sort(key=lambda x: x['p'])
                        return [x['c'] for x in elements]
                    
                    struct = build_tree(roots[0], token_dict)
                    with open(cfile, 'w', encoding='utf-8') as f: 
                        json.dump(struct, f, ensure_ascii=False)
                except: continue

            if struct:
                def count_leaves(node):
                    if isinstance(node, str): return 1
                    return sum(count_leaves(item) for item in node if not isinstance(item, dict))
                
                L_actual = count_leaves(struct)
                current_lang_models['Hierarchical']['x'].append(L_actual)
                current_lang_models['Hierarchical']['y'].append(calculate_open_nodes_theta(struct))
                current_lang_models['Branching']['x'].append(L_actual)
                current_lang_models['Branching']['y'].append(np.mean(simulate_branch_merge(L_actual)))
                current_lang_models['Flat']['x'].append(L_actual)
                current_lang_models['Flat']['y'].append(np.mean(simulate_flat_merge(L_actual)))

        print("\n    Generating visualization charts...") 

        h_y = current_lang_models['Hierarchical']['y']
        b_y = current_lang_models['Branching']['y']
        f_y = current_lang_models['Flat']['y']
        count = len(h_y)
        summary_stats.append({
            'name': name, 'count': count,
            'avg_h': np.mean(h_y) if count > 0 else 0,
            'avg_b': np.mean(b_y) if count > 0 else 0,
            'avg_f': np.mean(f_y) if count > 0 else 0
        })

        plot_single_language(name, code, lang_info['color'], current_lang_models)
        combined_results_for_plot.append({'name': name, 'color': lang_info['color'], 'models': current_lang_models})

    print("\n" + "="*85)
    print(f"{'Language':<12} | {'Sentences':<10} | {'Mean θ (Hier.)':<15} | {'Mean θ (Branch)':<15} | {'Mean θ (Flat)':<15}")
    print("-" * 85)
    for stat in summary_stats:
        print(f"{stat['name']:<12} | {stat['count']:<10} | {stat['avg_h']:<15.4f} | {stat['avg_b']:<15.4f} | {stat['avg_f']:<15.4f}")
    print("="*85)

    if combined_results_for_plot:
        print(">>> Generating cross-linguistic comparison figure...")
        plot_combined_24_lines_with_points(combined_results_for_plot)

    print("\n" + "="*100)
    print(f"{'Corpus/Lang':<18} | {'Valid Sents':<12} | {'Mean Length':<15} | {'Std Dev':<15}")
    print("-" * 100)
    
    all_lengths_combined = []
    
    for lang_res in combined_results_for_plot:
        # Extract sentence length lists from individual models
        lengths = np.array(lang_res['models']['Hierarchical']['x'])
        
        if len(lengths) > 0:
            count = len(lengths)
            avg_len = np.mean(lengths)
            std_len = np.std(lengths)
            
            # Print individual corpus stats
            print(f"{lang_res['name']:<18} | {count:<12} | {avg_len:<15.2f} | {std_len:<15.2f}")
            
            # Aggregate all length data
            all_lengths_combined.extend(lengths)
    
    # Print global aggregate stats
    if all_lengths_combined:
        all_lengths_combined = np.array(all_lengths_combined)
        total_count = len(all_lengths_combined)
        total_avg = np.mean(all_lengths_combined)
        total_std = np.std(all_lengths_combined)
        
        print("-" * 100)
        print(f"{'TOTAL (All Corpora)':<18} | {total_count:<12} | {total_avg:<15.2f} | {total_std:<15.2f}")
    
    print("="*100)

    print("\nTask completed. Please check the 'figures' folder.")