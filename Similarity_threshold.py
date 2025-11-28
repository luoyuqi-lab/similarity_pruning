import numpy as np
from Functions import txt_loader, cdtw
import time
from collections import defaultdict
import os
import csv
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

def analyze_similarity_thresholds(dataset_name, w):
    """
    使用LOOCV方式分析同类别和不同类别时间序列之间的CDTW距离
    
    Args:
        dataset_name: 数据集名称
        w: CDTW的warping window参数
    
    Returns:
        dict: 包含分析结果的字典
    """
    print(f"开始分析数据集: {dataset_name}")
    
    # 加载训练数据
    train_data = txt_loader(f'C:\\Users\\luoyu\\Desktop\\Univariate_arff\\{dataset_name}\\{dataset_name}_TRAIN.txt')
    
    # 按类别组织数据
    class_data = defaultdict(list)
    for i, template in enumerate(train_data):
        label = int(template[0])  # 第一列是标签
        time_series = template[1:]  # 其余列是时间序列数据
        class_data[label].append((i, time_series))
    
    print(f"数据集 {dataset_name} 包含 {len(class_data)} 个类别")
    for label, templates in class_data.items():
        print(f"  类别 {label}: {len(templates)} 个模板")
    
    # 存储距离统计
    same_class_distances = []
    different_class_distances = []
    
    # LOOCV: 每次留出一个样本作为测试，其余作为模板
    total_samples = len(train_data)
    processed = 0
    
    for test_idx in range(total_samples):
        test_template = train_data[test_idx]
        test_label = int(test_template[0])
        test_series = np.array(test_template[1:])
        
        # 计算与所有其他模板的CDTW距离
        for template_idx in range(total_samples):
            if template_idx == test_idx:  # 跳过自己
                continue
                
            template_template = train_data[template_idx]
            template_label = int(template_template[0])
            template_series = np.array(template_template[1:])
            
            # 计算CDTW距离
            try:
                distance = cdtw(template_series, test_series, w)
                
                # 根据类别关系分类距离
                if template_label == test_label:
                    same_class_distances.append(distance)
                else:
                    different_class_distances.append(distance)
                    
            except Exception as e:
                print(f"计算CDTW距离时出错: {e}")
                continue
        
        processed += 1
        if processed % 10 == 0:
            print(f"已处理 {processed}/{total_samples} 个模板")
    
    # 计算统计信息
    results = {
        'dataset_name': dataset_name,
        'w': w,
        'same_class': {
            'count': len(same_class_distances),
            'mean': np.mean(same_class_distances) if same_class_distances else 0,
            'std': np.std(same_class_distances) if same_class_distances else 0,
            'min': np.min(same_class_distances) if same_class_distances else 0,
            'max': np.max(same_class_distances) if same_class_distances else 0,
            'median': np.median(same_class_distances) if same_class_distances else 0
        },
        'different_class': {
            'count': len(different_class_distances),
            'mean': np.mean(different_class_distances) if different_class_distances else 0,
            'std': np.std(different_class_distances) if different_class_distances else 0,
            'min': np.min(different_class_distances) if different_class_distances else 0,
            'max': np.max(different_class_distances) if different_class_distances else 0,
            'median': np.median(different_class_distances) if different_class_distances else 0
        }
    }
    
    # 计算距离比率
    if results['different_class']['mean'] > 0:
        results['distance_ratio'] = results['same_class']['mean'] / results['different_class']['mean']
    else:
        results['distance_ratio'] = float('inf')
    
    # 添加原始距离数据用于可视化
    results['same_class_distances'] = same_class_distances
    results['different_class_distances'] = different_class_distances
    
    return results

def print_results(results):
    """打印分析结果"""
    print("\n" + "="*60)
    print(f"数据集: {results['dataset_name']}")
    print(f"Warping Window (w): {results['w']}")
    print("="*60)
    
    print("\n同类别时间序列CDTW距离统计:")
    print(f"  模板数量: {results['same_class']['count']}")
    print(f"  平均距离: {results['same_class']['mean']:.6f}")
    print(f"  标准差: {results['same_class']['std']:.6f}")
    print(f"  最小距离: {results['same_class']['min']:.6f}")
    print(f"  最大距离: {results['same_class']['max']:.6f}")
    print(f"  中位数: {results['same_class']['median']:.6f}")
    
    print("\n不同类别时间序列CDTW距离统计:")
    print(f"  模板数量: {results['different_class']['count']}")
    print(f"  平均距离: {results['different_class']['mean']:.6f}")
    print(f"  标准差: {results['different_class']['std']:.6f}")
    print(f"  最小距离: {results['different_class']['min']:.6f}")
    print(f"  最大距离: {results['different_class']['max']:.6f}")
    print(f"  中位数: {results['different_class']['median']:.6f}")
    
    print(f"\n距离比率 (同类别平均/不同类别平均): {results['distance_ratio']:.6f}")
    
    if results['distance_ratio'] < 1:
        print("✓ 同类别距离小于不同类别距离，CDTW能有效区分类别")
    else:
        print("✗ 同类别距离大于不同类别距离，CDTW区分效果不佳")



def save_results_to_csv(results, csv_filename="similarity_threshold_results.csv"):
    """将结果保存到CSV文件"""
    # 确保目录存在
    csv_dir = "C:\\Users\\luoyu\\Desktop\\KNN_DTW_tuning_result"
    if not os.path.exists(csv_dir):
        os.makedirs(csv_dir)
    
    csv_path = os.path.join(csv_dir, csv_filename)
    
    # 检查文件是否存在，如果不存在则创建并写入表头
    file_exists = os.path.exists(csv_path)
    
    with open(csv_path, mode='a', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        
        # 如果文件不存在，写入表头
        if not file_exists:
            writer.writerow([
                'Dataset', 'W', 'Same_Class_Count', 'Same_Class_Mean', 'Same_Class_Std', 
                'Same_Class_Min', 'Same_Class_Max', 'Same_Class_Median',
                'Different_Class_Count', 'Different_Class_Mean', 'Different_Class_Std',
                'Different_Class_Min', 'Different_Class_Max', 'Different_Class_Median',
                'Distance_Ratio', 'Effectiveness'
            ])
        
        # 写入数据行
        effectiveness = "Effective" if results['distance_ratio'] < 1 else "Ineffective"
        writer.writerow([
            results['dataset_name'],
            results['w'],
            results['same_class']['count'],
            f"{results['same_class']['mean']:.6f}",
            f"{results['same_class']['std']:.6f}",
            f"{results['same_class']['min']:.6f}",
            f"{results['same_class']['max']:.6f}",
            f"{results['same_class']['median']:.6f}",
            results['different_class']['count'],
            f"{results['different_class']['mean']:.6f}",
            f"{results['different_class']['std']:.6f}",
            f"{results['different_class']['min']:.6f}",
            f"{results['different_class']['max']:.6f}",
            f"{results['different_class']['median']:.6f}",
            f"{results['distance_ratio']:.6f}",
            effectiveness
        ])
    
    print(f"结果已保存到CSV文件: {csv_path}")

def visualize_distance_distribution(results, save_plot=True):
    """
    可视化同类别和不同类别距离的分布
    
    Args:
        results: 分析结果字典
        save_plot: 是否保存图片
    """
    # 设置中文字体
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    
    # 创建图形
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    dataset_name = results['dataset_name']
    same_distances = results['same_class_distances']
    different_distances = results['different_class_distances']
    
    # 设置颜色
    same_color = '#2E86AB'  # 蓝色
    diff_color = '#A23B72'  # 红色
    
    # 1. 直方图
    ax1.hist(same_distances, bins=30, alpha=0.7, color=same_color, 
             label=f'同类别 (n={len(same_distances)})', density=True)
    ax1.hist(different_distances, bins=30, alpha=0.7, color=diff_color, 
             label=f'不同类别 (n={len(different_distances)})', density=True)
    
    ax1.set_xlabel('CDTW距离')
    ax1.set_ylabel('密度')
    ax1.set_title(f'{dataset_name} - CDTW距离分布直方图')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. 箱线图
    data_to_plot = [same_distances, different_distances]
    labels = ['同类别', '不同类别']
    colors = [same_color, diff_color]
    
    bp = ax2.boxplot(data_to_plot, labels=labels, patch_artist=True)
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    ax2.set_ylabel('CDTW距离')
    ax2.set_title(f'{dataset_name} - CDTW距离箱线图')
    ax2.grid(True, alpha=0.3)
    
    # 添加统计信息
    same_mean = np.mean(same_distances)
    diff_mean = np.mean(different_distances)
    same_std = np.std(same_distances)
    diff_std = np.std(different_distances)
    
    # 在箱线图上添加均值线
    ax2.axhline(y=same_mean, color=same_color, linestyle='--', alpha=0.8, 
                label=f'同类别均值: {same_mean:.2f}')
    ax2.axhline(y=diff_mean, color=diff_color, linestyle='--', alpha=0.8, 
                label=f'不同类别均值: {diff_mean:.2f}')
    ax2.legend()
    
    plt.tight_layout()
    
    # 保存图片
    if save_plot:
        # 确保目录存在
        plot_dir = "C:\\Users\\luoyu\\Desktop\\KNN_DTW_tuning_result"
        if not os.path.exists(plot_dir):
            os.makedirs(plot_dir)
        
        plot_path = os.path.join(plot_dir, f"{dataset_name}_distance_distribution.png")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"距离分布图已保存到: {plot_path}")
    
    plt.show()

def create_detailed_visualization(results, save_plot=True):
    """
    创建更详细的可视化，包括距离区间统计
    
    Args:
        results: 分析结果字典
        save_plot: 是否保存图片
    """
    # 设置中文字体
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    
    dataset_name = results['dataset_name']
    same_distances = results['same_class_distances']
    different_distances = results['different_class_distances']
    
    # 创建图形
    fig = plt.figure(figsize=(16, 10))
    
    # 1. 距离区间统计
    ax1 = plt.subplot(2, 3, 1)
    
    # 计算距离区间
    all_distances = same_distances + different_distances
    min_dist = min(all_distances)
    max_dist = max(all_distances)
    
    # 创建10个等宽区间
    bins = np.linspace(min_dist, max_dist, 11)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    
    # 统计每个区间的数量
    same_counts, _ = np.histogram(same_distances, bins=bins)
    diff_counts, _ = np.histogram(different_distances, bins=bins)
    
    x = np.arange(len(bin_centers))
    width = 0.35
    
    ax1.bar(x - width/2, same_counts, width, label='同类别', alpha=0.7, color='#2E86AB')
    ax1.bar(x + width/2, diff_counts, width, label='不同类别', alpha=0.7, color='#A23B72')
    
    ax1.set_xlabel('距离区间')
    ax1.set_ylabel('模板数量')
    ax1.set_title(f'{dataset_name} - 距离区间分布')
    ax1.set_xticks(x)
    ax1.set_xticklabels([f'{bin_centers[i]:.1f}' for i in range(len(bin_centers))], rotation=45)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. 累积分布函数
    ax2 = plt.subplot(2, 3, 2)
    
    # 排序距离
    same_sorted = np.sort(same_distances)
    diff_sorted = np.sort(different_distances)
    
    # 计算累积分布
    same_cdf = np.arange(1, len(same_sorted) + 1) / len(same_sorted)
    diff_cdf = np.arange(1, len(diff_sorted) + 1) / len(diff_sorted)
    
    ax2.plot(same_sorted, same_cdf, label='同类别', color='#2E86AB', linewidth=2)
    ax2.plot(diff_sorted, diff_cdf, label='不同类别', color='#A23B72', linewidth=2)
    
    ax2.set_xlabel('CDTW距离')
    ax2.set_ylabel('累积概率')
    ax2.set_title(f'{dataset_name} - 累积分布函数')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. 距离比率分布
    ax3 = plt.subplot(2, 3, 3)
    
    # 计算每个距离的比率（相对于总体均值）
    overall_mean = np.mean(all_distances)
    same_ratios = np.array(same_distances) / overall_mean
    diff_ratios = np.array(different_distances) / overall_mean
    
    ax3.hist(same_ratios, bins=30, alpha=0.7, color='#2E86AB', 
             label=f'同类别 (均值={np.mean(same_ratios):.2f})', density=True)
    ax3.hist(diff_ratios, bins=30, alpha=0.7, color='#A23B72', 
             label=f'不同类别 (均值={np.mean(diff_ratios):.2f})', density=True)
    
    ax3.set_xlabel('距离比率 (相对于总体均值)')
    ax3.set_ylabel('密度')
    ax3.set_title(f'{dataset_name} - 距离比率分布')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. 统计摘要
    ax4 = plt.subplot(2, 3, 4)
    ax4.axis('off')
    
    # 创建统计摘要文本
    stats_text = f"""
    数据集: {dataset_name}
    
    同类别距离统计:
    • 模板数量: {len(same_distances)}
    • 平均距离: {np.mean(same_distances):.2f}
    • 标准差: {np.std(same_distances):.2f}
    • 最小距离: {np.min(same_distances):.2f}
    • 最大距离: {np.max(same_distances):.2f}
    • 中位数: {np.median(same_distances):.2f}
    
    不同类别距离统计:
    • 模板数量: {len(different_distances)}
    • 平均距离: {np.mean(different_distances):.2f}
    • 标准差: {np.std(different_distances):.2f}
    • 最小距离: {np.min(different_distances):.2f}
    • 最大距离: {np.max(different_distances):.2f}
    • 中位数: {np.median(different_distances):.2f}
    
    距离比率: {results['distance_ratio']:.4f}
    有效性: {'有效' if results['distance_ratio'] < 1 else '无效'}
    """
    
    ax4.text(0.05, 0.95, stats_text, transform=ax4.transAxes, fontsize=10,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
    
    # 5. 距离范围分析
    ax5 = plt.subplot(2, 3, 5)
    
    # 计算不同距离范围的模板数量
    ranges = [
        (0, np.percentile(all_distances, 25), '0-25%'),
        (np.percentile(all_distances, 25), np.percentile(all_distances, 50), '25-50%'),
        (np.percentile(all_distances, 50), np.percentile(all_distances, 75), '50-75%'),
        (np.percentile(all_distances, 75), np.percentile(all_distances, 100), '75-100%')
    ]
    
    same_range_counts = []
    diff_range_counts = []
    range_labels = []
    
    for low, high, label in ranges:
        same_count = sum(1 for d in same_distances if low <= d < high)
        diff_count = sum(1 for d in different_distances if low <= d < high)
        same_range_counts.append(same_count)
        diff_range_counts.append(diff_count)
        range_labels.append(label)
    
    x = np.arange(len(range_labels))
    width = 0.35
    
    ax5.bar(x - width/2, same_range_counts, width, label='同类别', alpha=0.7, color='#2E86AB')
    ax5.bar(x + width/2, diff_range_counts, width, label='不同类别', alpha=0.7, color='#A23B72')
    
    ax5.set_xlabel('距离百分位区间')
    ax5.set_ylabel('模板数量')
    ax5.set_title(f'{dataset_name} - 距离百分位分布')
    ax5.set_xticks(x)
    ax5.set_xticklabels(range_labels)
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    # 6. 重叠度分析
    ax6 = plt.subplot(2, 3, 6)
    
    # 计算重叠度
    same_mean = np.mean(same_distances)
    diff_mean = np.mean(different_distances)
    
    # 计算有多少同类别距离大于不同类别均值，以及多少不同类别距离小于同类别均值
    same_above_diff_mean = sum(1 for d in same_distances if d > diff_mean)
    diff_below_same_mean = sum(1 for d in different_distances if d < same_mean)
    
    overlap_data = [
        same_above_diff_mean,
        diff_below_same_mean,
        len(same_distances) - same_above_diff_mean,
        len(different_distances) - diff_below_same_mean
    ]
    
    overlap_labels = [
        f'同类别>异类均值\n({same_above_diff_mean})',
        f'异类别<同类均值\n({diff_below_same_mean})',
        f'同类别≤异类均值\n({len(same_distances) - same_above_diff_mean})',
        f'异类别≥同类均值\n({len(different_distances) - diff_below_same_mean})'
    ]
    
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']
    ax6.pie(overlap_data, labels=overlap_labels, colors=colors, autopct='%1.1f%%', startangle=90)
    ax6.set_title(f'{dataset_name} - 距离重叠度分析')
    
    plt.tight_layout()
    
    # 保存图片
    if save_plot:
        plot_dir = "C:\\Users\\luoyu\\Desktop\\KNN_DTW_tuning_result"
        if not os.path.exists(plot_dir):
            os.makedirs(plot_dir)
        
        plot_path = os.path.join(plot_dir, f"{dataset_name}_detailed_analysis.png")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"详细分析图已保存到: {plot_path}")
    
    plt.show()

def main():
    """主函数 - 可以分析多个数据集"""
    
    # 数据集配置 (数据集名称, w参数)
    datasets = [
        ("Car", 6)]

    #     ("CBF", 15),
    #     ("CricketX", 30),
    #     ("CricketY", 51),
    #     ("CricketZ", 15),
    #     ("ECG5000", 2),
    #     ("FaceAll", 4),
    #     ("FaceFour", 7),
    #     ("FacesUCR", 16),
    #     ("Fish", 17),
    #     ("FordA", 5),
    #     ("FordB", 5),
    #     ("Lightning2", 39),
    #     ("Lightning7", 3),
    #     ("MedicalImages", 16),
    #     ("Plane", 8),
    #     ("Trace", 1),
    #     ("TwoLeadECG", 2),
    #     ("Wafer", 58),
    #     ("Yoga", 30)
    # ]
    
    print("开始CDTW相似度阈值分析...")
    print("使用LOOCV方式计算同类别和不同类别时间序列的CDTW距离")
    
    for dataset_name, w in datasets:
        try:
            # 检查数据文件是否存在
            data_path = f'C:\\Users\\luoyu\\Desktop\\Univariate_arff\\{dataset_name}\\{dataset_name}_TRAIN.txt'
            if not os.path.exists(data_path):
                print(f"警告: 数据集 {dataset_name} 的训练文件不存在，跳过")
                continue
            
            # 分析数据集
            results = analyze_similarity_thresholds(dataset_name, w)
            
            # 打印结果
            print_results(results)
            
            # 保存结果到CSV文件
            save_results_to_csv(results)
            
            # 创建可视化
            print("\n正在生成可视化图表...")
            visualize_distance_distribution(results, save_plot=True)
            create_detailed_visualization(results, save_plot=True)
            
        except Exception as e:
            print(f"处理数据集 {dataset_name} 时出错: {e}")
            continue
    
    print("\n分析完成！结果已保存到CSV文件:")
    print("C:\\Users\\luoyu\\Desktop\\KNN_DTW_tuning_result\\similarity_threshold_results.csv")

if __name__ == "__main__":
    main()
