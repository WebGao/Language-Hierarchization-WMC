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
import random

# -------------------------- 1. 基础配置 --------------------------
LANGUAGES = [
    {"code": "en", "name": "English-3-4", "file": "original_sampled_data/eng_3-4.txt", "splitter": r"(?<=[.!?;:])\s+|\n\r", "color": "#F7DC6F"},
    {"code": "en", "name": "English-4-5", "file": "original_sampled_data/eng_4-5.txt", "splitter": r"(?<=[.!?;:])\s+|\n\r", "color": "#9370DB"},
    {"code": "en", "name": "English-5-6", "file": "original_sampled_data/eng_5-6.txt", "splitter": r"(?<=[.!?;:])\s+|\n\r", "color": "#2E8B57"},
    {"code": "en", "name": "English-6-7", "file": "original_sampled_data/eng_6-7.txt", "splitter": r"(?<=[.!?;:])\s+|\n\r", "color": "#FF8C00"},
    {"code": "en", "name": "English-7-8", "file": "original_sampled_data/eng_7-8.txt", "splitter": r"(?<=[.!?;:])\s+|\n\r", "color": "#8B4513"},
    {"code": "en", "name": "English-8-9", "file": "original_sampled_data/eng_8-9.txt", "splitter": r"(?<=[.!?;:])\s+|\n\r", "color": "#1D8BFF"},
    {"code": "en", "name": "English-9-10", "file": "original_sampled_data/eng_9-10.txt", "splitter": r"(?<=[.!?;:])\s+|\n\r", "color": "#FFA757"}
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

# -------------------------- 2. 核心计算函数 --------------------------

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

# -------------------------- 3. 绘图函数 (Nature Style) --------------------------

def plot_single_language(name, code, color, models_data):
    # 设置 Nature 风格比例，紧凑型画布
    fig, ax = plt.subplots(figsize=(4.5, 4), dpi=300)
    
    # 绘制人类工作记忆容量（WMC）区间，使用中性浅灰色
    ax.axhspan(WMC_MIN, WMC_MAX, color='#E6E6E6', alpha=0.4, label='Human WMC', zorder=0)
    
    configs = {
        'Hierarchical': {'color': color, 'ls': '-', 'marker': 'o', 'label': 'Hierarchical'},
        'Branching':    {'color': '#555555', 'ls': '--', 'marker': 'x', 'label': 'Linear Branching'}
    }
    
    for m_name, m_cfg in configs.items():
        data = models_data[m_name]
        x, y = np.array(data['x']), np.array(data['y'])
        if len(x) < 5: continue
        
        # 散点：减小尺寸，增加透明度以应对数据重叠
        ax.scatter(x, y, s=3, alpha=0.15, color=m_cfg['color'], edgecolors='none', zorder=2)
        
        try:
            # 曲线拟合
            xf = np.linspace(min(x), 65, 100)
            if m_name == 'Hierarchical':
                popt, _ = curve_fit(lambda t,a,b: a*np.log(t)+b, x, y)
                yf = popt[0]*np.log(xf) + popt[1]
            else:
                popt, _ = curve_fit(lambda t,a,b: a*t+b, x, y)
                yf = popt[0]*xf + popt[1]
            
            ax.plot(xf, yf, color=m_cfg['color'], ls=m_cfg['ls'], lw=1.5, label=m_cfg['label'], zorder=3)
        except: pass

    # 细节微调
    ax.set_title(f'{name}', fontsize=12, fontweight='bold', loc='left')
    ax.set_xlabel('Sentence Length ($L$)', fontsize=10)
    ax.set_ylabel(r'Cognitive Load ($\theta$)', fontsize=10)
    
    # 移除顶部和右侧边框
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    ax.set_xlim(2, 15)
    ax.set_ylim(0, 15)
    ax.legend(frameon=False, fontsize=8, loc='upper left')
    
    # 极简留白处理
    plt.tight_layout(pad=0.5)
    plt.savefig(os.path.join(FIGURE_OUTPUT_DIR, f"analysis_{name}.pdf"), bbox_inches='tight')
    plt.savefig(os.path.join(FIGURE_OUTPUT_DIR, f"analysis_{name}.png"), bbox_inches='tight')
    plt.close()

def plot_combined_24_lines_with_points(all_lang_data):
    # 仿 Nature 大图，双栏宽度
    fig, ax = plt.subplots(figsize=(10, 7), dpi=300)
    
    # 绘制人类工作记忆容量（WMC）区间
    ax.axhspan(WMC_MIN, WMC_MAX, color='#E6E6E6', alpha=0.4, zorder=0, label='Human WMC')
    
    xf = np.linspace(2, 25, 100) 
    model_curves = {'Hierarchical': [], 'Branching': []}
    
    # 1. 绘制每个语言的点和拟合线
    for lang in all_lang_data:
        name = lang['name']
        lang_color = lang['color']
        
        # 用于控制 Legend 只对每个语言显示一次
        label_added = False 
        
        for m_name, fit_type in [('Hierarchical', 'log'), ('Branching', 'linear')]:
            data = lang['models'][m_name]
            x, y = np.array(data['x']), np.array(data['y'])
            if len(x) < 5: continue
            
            # --- 绘制原始点 (参考 plot_single_language) ---
            # s=1, alpha=0.05 以应对 24 种语言叠加的巨大数据量
            ax.scatter(x, y, s=20, alpha=0.05, color=lang_color, edgecolors='none', zorder=1)
            
            try:
                if fit_type == 'log':
                    popt, _ = curve_fit(lambda t,a,b: a*np.log(t)+b, x, y)
                    yf = popt[0]*np.log(xf) + popt[1]
                    ls = '-'
                else:
                    popt, _ = curve_fit(lambda t,a,b: a*t+b, x, y)
                    yf = popt[0]*xf + popt[1]
                    ls = ':' # Branching 线型设为点状以示区别
                
                model_curves[m_name].append(yf)
                
                # 设置 Legend
                current_label = name if not label_added else None
                ax.plot(xf, yf, color=lang_color, ls=ls, lw=1.0, alpha=0.4, label=current_label, zorder=2)
                label_added = True
            except: pass

    # 2. 绘制全局平均粗线 (使用高对比度学术配色)
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
                
    # 细节美化
    ax.set_title('Cross-linguistic Complexity Profile (Global Analysis)', fontsize=14, fontweight='bold', loc='left')
    ax.set_xlabel('Sentence Length ($L$)', fontsize=11)
    ax.set_ylabel(r'Processing Load ($\theta$)', fontsize=11)
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # 坐标范围固定在需求区间
    ax.set_xlim(2, 25)
    ax.set_ylim(0, 15)
    
    # 图例配置：放在右侧
    ax.legend(frameon=False, loc='upper left', bbox_to_anchor=(1.02, 1), fontsize=8, ncol=1)
    
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURE_OUTPUT_DIR, "analysis_combined_nature.pdf"), bbox_inches='tight')
    plt.savefig(os.path.join(FIGURE_OUTPUT_DIR, "analysis_combined_nature.png"), bbox_inches='tight')
    plt.close()
    
# -------------------------- 4. 主程序逻辑 --------------------------

if __name__ == "__main__":
    combined_results_for_plot = []
    summary_stats = []
    
    # 固定随机种子
    random.seed(42)
    SAMPLE_SIZE = 5000
    
    print(f"{'='*20} 跨语言认知负荷分析启动 {'='*20}")
    
    for lang_info in LANGUAGES:
        code, name = lang_info['code'], lang_info['name']
        if not os.path.exists(lang_info['file']):
            print(f"警告: 未找到 {name} 的语料文件: {lang_info['file']}")
            continue
            
        print(f"\n>>> 正在处理语言: {name} ({code})") 
        
        content = None
        for enc in ['utf-8', 'gbk', 'utf-8-sig', 'latin-1']:
            try:
                with open(lang_info['file'], 'r', encoding=enc) as f:
                    content = f.read()
                break
            except: continue
        if content is None: continue

        # 初始化 Stanza 管道
        nlp = stanza.Pipeline(lang=code, processors='tokenize,pos,lemma,depparse', 
                              dir=STANZA_MODEL_DIR, logging_level='ERROR')

        # 原始句子切分
        raw_sents_all = re.split(lang_info['splitter'], content)
        raw_sents_all = [s.strip() for s in raw_sents_all if s.strip()]
        
        # --- 随机采样 3000 个句子 ---
        if len(raw_sents_all) > SAMPLE_SIZE:
            raw_sents = random.sample(raw_sents_all, SAMPLE_SIZE)
            print(f"    [采样] 已从 {len(raw_sents_all)} 句中随机抽取 {SAMPLE_SIZE} 句")
        else:
            raw_sents = raw_sents_all
            print(f"    [采样] 语料不足 {SAMPLE_SIZE} 句，全量处理 ({len(raw_sents)} 句)")

        temp_data = []
        print(f"    [阶段 1/3] 预分词与初步筛选...")
        for s in raw_sents:
            s = s.replace('\u3000', ' ').strip()
            if not s: continue
            
            if code in ["zh-hans", "ja"]:
                doc_pre = nlp(s)
                words = [w.text for sent in doc_pre.sentences for w in sent.words if not re.match(r'^[\W\d]+$', w.text.strip())]
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
        
        print(f"    [统计信息] 句子长度分布:")
        print(f"        采样后样本数: {len(lens)}")
        print(f"        均值 (Mean): {mean_len:.2f}, 标准差 (Std): {std_len:.2f}")
        print(f"        Q1 (25%): {q1:.2f}, Q3 (75%): {q3:.2f}, IQR: {iqr:.2f}")
        print(f"        异常值过滤下界 (Lower Bound): {lower:.2f}")
        print(f"        异常值过滤上界 (Upper Bound): {upper:.2f}")
        
        filtered_entries = [d for d in temp_data if lower <= d['len'] <= upper]
        total_to_process = len(filtered_entries)
        print(f"    [阶段 2/3] 过滤后待处理句子总数: {total_to_process}")

        current_lang_models = {
            'Hierarchical': {'x':[],'y':[]}, 
            'Branching': {'x':[],'y':[]}, 
            'Flat': {'x':[],'y':[]},
            'Color': lang_info['color']
        }
        
        STRUCTURE_CACHE_DIR_LANG = f"./structure_cache_{code}"
        os.makedirs(STRUCTURE_CACHE_DIR_LANG, exist_ok=True)

        print(f"    [阶段 3/3] 依存分析与复杂度计算:")
        for idx, entry in enumerate(filtered_entries):
            if (idx + 1) % 5 == 0 or (idx + 1) == total_to_process:
                print(f"\r        处理进度: {idx+1}/{total_to_process} ({(idx+1)/total_to_process:.1%})", end="", flush=True)
            
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

        print("\n    正在生成可视化图表...") 

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
        print(">>> 正在生成多语言对比总图...")
        plot_combined_24_lines_with_points(combined_results_for_plot)

    # -------------------------- 5. 完整：年龄段趋势分析图 (含 F, R^2, p 统计) --------------------------
    if combined_results_for_plot:
        print(">>> 正在生成年龄段均值趋势对比图并执行统计分析...")
        from scipy import stats  # 用于执行标准回归统计
        
        ages = []
        means = []
        sems = []
        
        for item in combined_results_for_plot:
            h_y = np.array(item['models']['Hierarchical']['y'])
            if len(h_y) == 0: continue
            
            # 从名称中解析年龄中值
            match = re.search(r'(\d+)-(\d+)', item['name'])
            if match:
                age_start, age_end = map(int, match.groups())
                age_mid = (age_start + age_end) / 2.0
            else:
                age_mid = len(ages) + 1 
            
            # 计算指标
            avg = np.mean(h_y)
            sem = np.std(h_y) / np.sqrt(len(h_y)) 
            
            ages.append(age_mid)
            means.append(avg)
            sems.append(sem)
            
        ages = np.array(ages)
        means = np.array(means)
        sems = np.array(sems)

        sort_idx = np.argsort(ages)
        ages, means, sems = ages[sort_idx], means[sort_idx], sems[sort_idx]

        fig, ax = plt.subplots(figsize=(6, 5), dpi=300)

        # 1. 使用灰色阴影展示误差 (Standard Error of the Mean)
        ax.fill_between(ages, means - sems, means + sems, color='gray', alpha=0.2, label='SEM Shadow', zorder=1)
        
        # 绘制均值观测点
        ax.scatter(ages, means, color='#FF8C00', edgecolors='white', s=50, label='Mean $\theta$ (Observed)', zorder=3)

        # 2. 对数拟合与统计计算 (Log-linear Regression)
        try:
            # 执行线性回归：y = a * log(x) + b
            x_log = np.log(ages)
            slope, intercept, r_value, p_value, std_err = stats.linregress(x_log, means)
            
            # 计算关键统计结果
            r_squared = r_value**2
            n = len(ages)
            k = 1  # 自变量个数
            # F = (R^2 / k) / ((1 - R^2) / (n - k - 1))
            f_statistic = (r_squared / k) / ((1 - r_squared) / (n - k - 1)) if r_squared < 1 else float('inf')

            # 打印统计报告
            print("\n" + "="*40)
            print("【回归分析统计结果】")
            print(f"回归方程: θ = {slope:.4f} * ln(Age) + {intercept:.4f}")
            print(f"决定系数 R²: {r_squared:.4f}")
            print(f"F 统计量: {f_statistic:.4f}")
            print(f"显著性 p-value: {p_value:.4e}")
            print("="*40 + "\n")

            # 拟合曲线绘图
            x_fit = np.linspace(min(ages)*0.9, max(ages)*1.1, 100)
            y_fit = slope * np.log(x_fit) + intercept
            
            # 在图例中展示 R2 和 p
            ax.plot(x_fit, y_fit, color='#FF8C00', lw=2.5, linestyle='-', 
                    label=f'Log Fit ($R^2={r_squared:.3f}, p={p_value:.2e}$)', zorder=2)
                    
        except Exception as e:
            print(f"拟合统计失败: {e}")

        # 细节美化 (保持原有设置)
        ax.set_title('Cognitive Load Development Trend', fontsize=12, fontweight='bold', loc='left')
        ax.set_xlabel('Age (Midpoint of Group)', fontsize=10)
        ax.set_ylabel(r'Mean Cognitive Load ($\theta$)', fontsize=10)
        
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.grid(axis='y', linestyle='--', alpha=0.3)
        ax.legend(frameon=False, fontsize=9)

        plt.tight_layout()
        plt.savefig(os.path.join(FIGURE_OUTPUT_DIR, "age_trend_analysis.png"), bbox_inches='tight')
        plt.close()
        print(">>> 趋势图已保存至 figures 文件夹。")

    print("\n" + "="*100)
    print(f"{'语料名称':<18} | {'有效句子数':<12} | {'长度均值(Mean)':<15} | {'长度标准差(Std)':<15}")
    print("-" * 100)
    
    all_lengths_list = []
    
    for item in combined_results_for_plot:
        # 提取当前语料的所有句子长度 (x轴数据)
        current_lengths = np.array(item['models']['Hierarchical']['x'])
        
        if len(current_lengths) > 0:
            count = len(current_lengths)
            avg_len = np.mean(current_lengths)
            std_len = np.std(current_lengths)
            
            # 打印单个语料的统计
            print(f"{item['name']:<18} | {count:<12} | {avg_len:<15.2f} | {std_len:<15.2f}")
            
            # 汇总到全局列表
            all_lengths_list.extend(current_lengths)
    
    # 计算并打印全局汇总
    if all_lengths_list:
        all_lengths_np = np.array(all_lengths_list)
        total_cnt = len(all_lengths_np)
        total_avg = np.mean(all_lengths_np)
        total_std = np.std(all_lengths_np)
        
        print("-" * 100)
        print(f"{'TOTAL (所有语料汇总)':<18} | {total_cnt:<12} | {total_avg:<15.2f} | {total_std:<15.2f}")
    
    print("="*100)

    print("\n任务完成。请检查 'figures' 文件夹。")

# if __name__ == "__main__":
#     combined_results_for_plot = []
#     summary_stats = []
    
#     # 固定随机种子
#     random.seed(20)
#     SAMPLE_SIZE = 10000
    
#     print(f"{'='*20} 跨语言认知负荷分析启动 {'='*20}")
    
#     for lang_info in LANGUAGES:
#         code, name = lang_info['code'], lang_info['name']
#         if not os.path.exists(lang_info['file']):
#             print(f"警告: 未找到 {name} 的语料文件: {lang_info['file']}")
#             continue
            
#         print(f"\n>>> 正在处理语言: {name} ({code})") 
        
#         content = None
#         for enc in ['utf-8', 'gbk', 'utf-8-sig', 'latin-1']:
#             try:
#                 with open(lang_info['file'], 'r', encoding=enc) as f:
#                     content = f.read()
#                 break
#             except: continue
#         if content is None: continue

#         # 初始化 Stanza 管道
#         nlp = stanza.Pipeline(lang=code, processors='tokenize,pos,lemma,depparse', 
#                               dir=STANZA_MODEL_DIR, logging_level='ERROR')

#         # 原始句子切分
#         raw_sents_all = re.split(lang_info['splitter'], content)
#         raw_sents_all = [s.strip() for s in raw_sents_all if s.strip()]
        
#         # --- 采样逻辑 (不修改) ---
#         if len(raw_sents_all) > SAMPLE_SIZE:
#             raw_sents = random.sample(raw_sents_all, SAMPLE_SIZE)
#         else:
#             raw_sents = raw_sents_all

#         temp_data = []
#         for s in raw_sents:
#             s = s.replace('\u3000', ' ').strip()
#             if not s: continue
#             if code in ["zh-hans", "ja"]:
#                 doc_pre = nlp(s)
#                 words = [w.text for sent in doc_pre.sentences for w in sent.words if not re.match(r'^[\W\d]+$', w.text.strip())]
#             else:
#                 clean_text = re.sub(r'[^a-zA-Z0-9àáâãäåçèéêëìíîïñòóôõöùúûüýÿА-Яа-я]', ' ', s)
#                 words = [word for word in clean_text.split() if word]
#             L = len(words)
#             if L > 0: 
#                 clean_text = "".join(words) if code in ['zh-hans', 'ja'] else " ".join(words)
#                 temp_data.append({'text': clean_text, 'len': L, 'word_list': words})
        
#         if not temp_data: continue
#         lens = [d['len'] for d in temp_data]
#         q1, q3 = np.percentile(lens, 25), np.percentile(lens, 75)
#         iqr = q3 - q1
#         lower, upper = max(3, q1 - 1.5*iqr), q3 + 1.5*iqr
        
#         filtered_entries = [d for d in temp_data if lower <= d['len'] <= upper]
#         current_lang_models = {'Hierarchical': {'x':[],'y':[]}, 'Branching': {'x':[],'y':[]}, 'Flat': {'x':[],'y':[]}}
        
#         # 依存分析与缓存 (不修改)
#         STRUCTURE_CACHE_DIR_LANG = f"./structure_cache_{code}"
#         os.makedirs(STRUCTURE_CACHE_DIR_LANG, exist_ok=True)

#         for idx, entry in enumerate(filtered_entries):
#             txt = entry['text']
#             ckey = hashlib.sha256(f"{code}:{txt}".encode('utf-8')).hexdigest()
#             cfile = os.path.join(STRUCTURE_CACHE_DIR_LANG, f"{ckey}.json")
#             struct = None
#             if os.path.exists(cfile):
#                 with open(cfile, 'r', encoding='utf-8') as f: struct = json.load(f)
#             else:
#                 try:
#                     doc = nlp(txt)
#                     if not doc.sentences: continue
#                     sent = doc.sentences[0]
#                     token_dict = {w.id: {'text': w.text, 'head': w.head} for w in sent.words}
#                     roots = [i for i, t in token_dict.items() if t['head'] == 0]
#                     def build_tree(idx, d):
#                         token = d[idx]; child_ids = [k for k, t in d.items() if t['head'] == idx]
#                         if not child_ids: return token['text']
#                         elements = [{'p': idx, 'c': token['text']}]
#                         for cid in child_ids: elements.append({'p': cid, 'c': build_tree(cid, d)})
#                         elements.sort(key=lambda x: x['p']); return [x['c'] for x in elements]
#                     struct = build_tree(roots[0], token_dict)
#                     with open(cfile, 'w', encoding='utf-8') as f: json.dump(struct, f, ensure_ascii=False)
#                 except: continue
#             if struct:
#                 def count_leaves(node):
#                     if isinstance(node, str): return 1
#                     return sum(count_leaves(item) for item in node if not isinstance(item, dict))
#                 L_actual = count_leaves(struct)
#                 current_lang_models['Hierarchical']['x'].append(L_actual)
#                 current_lang_models['Hierarchical']['y'].append(calculate_open_nodes_theta(struct))

#         combined_results_for_plot.append({'name': name, 'color': lang_info['color'], 'models': current_lang_models})

#     # --- 新增：打印每个语料的统计信息 ---
#     print("\n" + "="*85)
#     print(f"{'语料名称':<15} | {'句子数量':<10} | {'词数均值':<10} | {'词数标准差':<10} | {'Theta均值':<10}")
#     print("-" * 85)
#     for item in combined_results_for_plot:
#         h_x = np.array(item['models']['Hierarchical']['x'])
#         h_y = np.array(item['models']['Hierarchical']['y'])
#         if len(h_x) > 0:
#             print(f"{item['name']:<15} | {len(h_x):<10} | {np.mean(h_x):<10.2f} | {np.std(h_x):<10.2f} | {np.mean(h_y):<10.4f}")
#     print("="*85)

#     # -------------------------- 5. 趋势分析图 (高画布 + 公式 + 显著性) --------------------------
#     if combined_results_for_plot:
#         from scipy import stats
#         print("\n>>> 正在生成增强型趋势分析图...")
        
#         ages_list, means_list, sems_list = [], [], []
#         for item in combined_results_for_plot:
#             data_y = np.array(item['models']['Hierarchical']['y'])
#             if len(data_y) == 0: continue
#             match = re.search(r'(\d+)-(\d+)', item['name'])
#             a_mid = (int(match.group(1)) + int(match.group(2))) / 2.0 if match else (len(ages_list)+1)
#             ages_list.append(a_mid); means_list.append(np.mean(data_y))
#             sems_list.append(np.std(data_y) / np.sqrt(len(data_y)))
        
#         idx_sort = np.argsort(ages_list)
#         x_data, y_data, e_data = np.array(ages_list)[idx_sort], np.array(means_list)[idx_sort], np.array(sems_list)[idx_sort]

#         fig, ax = plt.subplots(figsize=(6, 7), dpi=300)
        
#         ax.fill_between(x_data, y_data - e_data, y_data + e_data, color='#D3D3D3', alpha=0.4, label='Data SEM', zorder=1)
#         ax.scatter(x_data, y_data, color='#FF8C00', s=50, edgecolors='white', zorder=4, label='Age Group Mean')

#         try:
#             def log_func(x, a, b): return a * np.log(x) + b
#             popt, pcov = curve_fit(log_func, x_data, y_data)
#             x_fit = np.linspace(min(x_data), max(x_data), 100)
#             y_fit = log_func(x_fit, *popt)
            
#             r_val, p_val = stats.pearsonr(np.log(x_data), y_data)
            
#             perr = np.sqrt(np.diag(pcov))
#             ax.plot(x_fit, y_fit, color='#FF8C00', lw=2.5, label='Log Regression', zorder=5)
#             ax.fill_between(x_fit, log_func(x_fit, popt[0]-perr[0], popt[1]-perr[1]), 
#                            log_func(x_fit, popt[0]+perr[0], popt[1]+perr[1]), 
#                            color='#FF8C00', alpha=0.15, zorder=2)

#             significance = "p < 0.001" if p_val < 0.001 else f"p = {p_val:.3f}"
#             formula_text = f"$y = {popt[0]:.2f} \cdot \ln(x) + {popt[1]:.2f}$\n$R = {r_val:.3f}, {significance}$"
#             ax.text(0.05, 0.92, formula_text, transform=ax.transAxes, fontsize=10, 
#                     verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))
#         except: pass

#         ax.set_title('Cognitive Load Growth Analysis', fontsize=14, fontweight='bold', loc='left', pad=20)
#         ax.set_xlabel('Age (Years)', fontsize=11); ax.set_ylabel(r'Cognitive Load ($\theta$)', fontsize=11)
#         ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
#         ax.grid(axis='y', linestyle=':', alpha=0.6)
#         ax.legend(frameon=False, loc='lower right', fontsize=9)

#         plt.tight_layout()
#         plt.savefig(os.path.join(FIGURE_OUTPUT_DIR, "enhanced_trend_analysis.png"), bbox_inches='tight')
#         plt.close()
#         print(">>> 增强型趋势分析图已生成。")

#     print("\n任务完成。")
    