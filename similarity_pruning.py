import numpy as np
from Functions import (
    txt_loader,
    load_similarity_statistics,
    KNN_DTW_original,
    KNN_DTW_with_pruning,
    save_pruning_results,
)
import time

from collections import defaultdict

from collections import defaultdict

def print_results(results):
    """打印结果"""
    print("\n" + "="*60)
    print(f"数据集: {results['dataset_name']}")
    print(f"方法: {results['method']}")
    print(f"参数: w={results['w']}, k={results['k']}")
    
    if results['method'] == 'Pruning':
        print(f"同类别距离中位数: {results['same_class_median']:.6f}")
        print(f"剪枝系数: {results['pruning_coefficient']}")
        print(f"剪枝阈值: {results['pruning_threshold']:.6f}")
        print(f"早期停止阈值: {results['early_stop_count']} 个距离")
    
    print("="*60)
    
    print(f"\n分类结果:")
    print(f"  准确率: {results['accuracy']:.6f} ({results['accuracy']*100:.2f}%)")
    print(f"  总时间: {results['total_time']:.2f} 秒")
    
    if results['method'] == 'Pruning':
        print(f"  剪枝率: {results['pruning_rate']:.6f} ({results['pruning_rate']*100:.2f}%)")
        print(f"  加速比: {results['speedup_factor']:.2f}x")
        
        if results['pruning_rate'] > 0:
            print(f"\n剪枝效果分析:")
            print(f"  • 有 {results['pruning_rate']*100:.1f}% 的测试样本触发了早期停止")
            print(f"  • 平均加速比: {results['speedup_factor']:.2f}x")
    
    print("="*60)


def estimate_baseline_time(dataset_name, w, k):
    """
    估算基准时间（不使用剪枝的完整KNN-DTW时间）
    这里使用一个简化的估算方法
    """
    # 加载数据获取规模信息
    train_data = txt_loader(f'C:\\Users\\luoyu\\Desktop\\Univariate_arff\\{dataset_name}\\{dataset_name}_TRAIN.txt')
    test_data = txt_loader(f'C:\\Users\\luoyu\\Desktop\\Univariate_arff\\{dataset_name}\\{dataset_name}_TEST.txt')
    
    # 基于数据规模估算基准时间（这是一个粗略估算）
    train_size = len(train_data)
    test_size = len(test_data)
    
    # 假设每个CDTW计算需要0.001秒（这个值需要根据实际情况调整）
    estimated_cdtw_time = 0.001
    baseline_time = train_size * test_size * estimated_cdtw_time
    
    return baseline_time

def main():
    """主函数"""
    
    # 数据集配置 (数据集名称, w参数, k参数)
    datasets = [
        ("Car", 6, 1)
    ]
    
    # 剪枝参数配置
    PRUNING_COEFFICIENT = 0.05  # 剪枝阈值系数，可调整此参数
    EARLY_STOP_COUNT = 2     # 早期停止所需的最小距离数量，可调整此参数
    
    print("开始KNN-DTW实验...")
    print("比较原始KNN-DTW和剪枝KNN-DTW的性能")
    print(f"剪枝参数: 系数={PRUNING_COEFFICIENT}, 早期停止阈值={EARLY_STOP_COUNT}")
    
    for dataset_name, w, k in datasets:
        try:
            print(f"\n{'='*60}")
            print(f"处理数据集: {dataset_name}")
            print(f"{'='*60}")
            
            # 1. 运行原始KNN-DTW分类
            print("\n1. 运行原始KNN-DTW分类...")
            original_accuracy, original_time = KNN_DTW_original(dataset_name, w, k)
            
            # 整理原始结果
            original_results = {
                'dataset_name': dataset_name,
                'method': 'Original',
                'w': w,
                'k': k,
                'accuracy': original_accuracy,
                'total_time': original_time
            }
            
            # 打印原始结果
            print_results(original_results)
            
            # 保存原始结果
            save_pruning_results(original_results)
            
            # 2. 运行剪枝KNN-DTW分类
            print("\n2. 运行剪枝KNN-DTW分类...")
            
            # 加载同类别距离中位数
            same_class_median = load_similarity_statistics(dataset_name, w)
            if same_class_median is None:
                print(f"跳过剪枝实验：无法获取统计信息")
                continue
            
            # 运行带剪枝的KNN-DTW分类
            pruning_accuracy, pruning_time, pruning_rate = KNN_DTW_with_pruning(
                dataset_name, w, k, same_class_median, pruning_coefficient=PRUNING_COEFFICIENT, early_stop_count=EARLY_STOP_COUNT
            )
            
            # 计算加速比（相对于原始方法）
            speedup_factor = original_time / pruning_time if pruning_time > 0 else 1.0
            
            # 整理剪枝结果
            pruning_results = {
                'dataset_name': dataset_name,
                'method': 'Pruning',
                'w': w,
                'k': k,
                'same_class_median': same_class_median,
                'pruning_coefficient': PRUNING_COEFFICIENT,
                'pruning_threshold': PRUNING_COEFFICIENT * same_class_median,
                'early_stop_count': EARLY_STOP_COUNT,
                'accuracy': pruning_accuracy,
                'total_time': pruning_time,
                'pruning_rate': pruning_rate,
                'speedup_factor': speedup_factor
            }
            
            # 打印剪枝结果
            print_results(pruning_results)
            
            # 保存剪枝结果
            save_pruning_results(pruning_results)
            
            # 3. 比较结果
            print(f"\n{'='*60}")
            print(f"性能比较 (数据集: {dataset_name})")
            print(f"{'='*60}")
            print(f"准确率比较:")
            print(f"  原始KNN-DTW: {original_accuracy:.6f} ({original_accuracy*100:.2f}%)")
            print(f"  剪枝KNN-DTW: {pruning_accuracy:.6f} ({pruning_accuracy*100:.2f}%)")
            print(f"  准确率变化: {pruning_accuracy - original_accuracy:+.4f} ({((pruning_accuracy/original_accuracy)-1)*100:+.2f}%)")
            
            print(f"\n时间比较:")
            print(f"  原始KNN-DTW: {original_time:.2f} 秒")
            print(f"  剪枝KNN-DTW: {pruning_time:.2f} 秒")
            print(f"  加速比: {speedup_factor:.2f}x")
            print(f"  时间节省: {((original_time-pruning_time)/original_time)*100:.1f}%")
            
            if pruning_rate > 0:
                print(f"\n剪枝效果:")
                print(f"  剪枝率: {pruning_rate:.6f} ({pruning_rate*100:.2f}%)")
                print(f"  剪枝阈值: {PRUNING_COEFFICIENT * same_class_median:.6f}")
            
            print(f"{'='*60}")
            
        except Exception as e:
            print(f"处理数据集 {dataset_name} 时出错: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    print("\n实验完成！结果已保存到:")
    print("C:\\Users\\luoyu\\Desktop\\KNN_DTW_tuning_result\\pruning_results.csv")

if __name__ == "__main__":
    main()
