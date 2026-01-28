import pandas as pd
from collections import defaultdict, Counter
import math

class MatrixPlannerOptimized:
    def __init__(self):
        # 主属性集合
        self.main_attrs = ['力量', '敏捷', '智识', '意志', '主能力']
        
        # 区域成本权重
        self.region_cost = {
            '枢纽区': 1,
            '源石研究园': 1,
            '矿脉源区': 1,
            '供能高地': 1,
            '武陵城': 7
        }
        
        # 加载数据
        self.load_data()
        
    def load_data(self):
        """加载武器属性数据"""
        # 读取test.txt中的武器名称
        try:
            with open('test.txt', 'r', encoding='utf-8') as f:
                test_weapon_names = [line.strip() for line in f if line.strip()]
            print(f"从test.txt中读取了 {len(test_weapon_names)} 个武器名称")
        except Exception as e:
            print(f"读取test.txt失败: {e}")
            test_weapon_names = []
        
        # 加载武器属性
        self.weapons = []
        if test_weapon_names:
            try:
                # 读取wqsx.txt，查找对应武器的属性
                df_weapons = pd.read_csv('wqsx.txt', sep='\t', encoding='utf-8')
                print(f"从wqsx.txt中找到了 {len(df_weapons)} 行武器数据")
                
                # 创建一个武器名称到属性的映射
                weapon_name_map = {}
                for _, row in df_weapons.iterrows():
                    weapon_name = row['名称']
                    weapon_name_map[weapon_name] = {
                        'name': weapon_name,
                        'main_attr': row['主属性'],
                        'add_attr': row['附加属性'],
                        'skill': row['技能'],
                        'category': row['类别'],
                        'rarity': row['稀有度']
                    }
                
                # 根据test.txt中的武器名称获取属性
                found_count = 0
                not_found_weapons = []
                
                for weapon_name in test_weapon_names:
                    if weapon_name in weapon_name_map:
                        self.weapons.append(weapon_name_map[weapon_name])
                        found_count += 1
                    else:
                        not_found_weapons.append(weapon_name)
                
                print(f"成功匹配了 {found_count} 个武器的属性")
                if not_found_weapons:
                    print(f"以下武器在wqsx.txt中未找到: {', '.join(not_found_weapons)}")
                    
            except Exception as e:
                print(f"加载武器数据失败: {e}")
                print("请确保wqsx.txt文件格式正确，包含以下列：类别、稀有度、名称、主属性、附加属性、技能")
        
        # 如果没有从文件中读取到数据，使用示例数据
        if not self.weapons:
            print("使用示例数据")
            self.weapons = [
                {'name': '钢铁余音', 'main_attr': '敏捷', 'add_attr': '物理', 'skill': '巧技'},
                {'name': '坚城铸造者', 'main_attr': '智识', 'add_attr': '终结技', 'skill': '昂扬'}
            ]
        
        # 加载区域属性
        self.regions = {}
        try:
            df_regions = pd.read_csv('jz.txt', sep='\t', encoding='utf-8')
            for _, row in df_regions.iterrows():
                region_name = row['区域']
                add_attrs = [row[f'附加属性{i}'] for i in range(1, 9)]
                skills = [row[f'技能{i}'] for i in range(1, 9)]
                self.regions[region_name] = {
                    'add_attrs': add_attrs,
                    'skills': skills
                }
            print(f"加载了 {len(self.regions)} 个区域的数据")
        except Exception as e:
            print(f"加载区域数据失败: {e}")
            print("使用示例区域数据")
            # 使用示例数据
            self.regions = {
                '枢纽区': {
                    'add_attrs': ['攻击', '灼热', '电磁', '寒冷', '自然', '源石', '终结技', '法术'],
                    'skills': ['强攻', '压制', '追袭', '粉碎', '巧技', '迸发', '流转', '效益']
                },
                '源石研究园': {
                    'add_attrs': ['攻击', '物理', '电磁', '寒冷', '自然', '暴击率', '终结技', '法术'],
                    'skills': ['压制', '追袭', '昂扬', '巧技', '附术', '医疗', '切骨', '效益']
                },
                '矿脉源区': {
                    'add_attrs': ['生命', '物理', '灼热', '寒冷', '自然', '暴击率', '源石', '治疗'],
                    'skills': ['强攻', '压制', '巧技', '残暴', '附术', '迸发', '夜幂', '效益']
                },
                '供能高地': {
                    'add_attrs': ['攻击', '生命', '物理', '灼热', '自然', '暴击率', '源石', '治疗'],
                    'skills': ['追袭', '粉碎', '昂扬', '残暴', '附术', '医疗', '切骨', '流转']
                },
                '武陵城': {
                    'add_attrs': ['攻击', '生命', '电磁', '寒冷', '暴击率', '终结技', '法术', '治疗'],
                    'skills': ['强攻', '粉碎', '残暴', '医疗', '切骨', '迸发', '夜幂', '流转']
                }
            }
    
    def find_region_for_group(self, weapons_group):
        """
        为武器组找到合适的区域
        返回 (区域, 需要固定的属性类型, 需要固定的属性值)
        """
        best_region = None
        best_fixed_type = None
        best_fixed_value = None
        best_score = float('inf')
        
        # 统计武器组的属性
        main_attrs_set = set(w['main_attr'] for w in weapons_group)
        add_attrs_set = set(w['add_attr'] for w in weapons_group)
        skills_set = set(w['skill'] for w in weapons_group)
        
        # 尝试每个区域
        for region_name, region_data in self.regions.items():
            region_add_attrs = set(region_data['add_attrs'])
            region_skills = set(region_data['skills'])
            
            # 检查所有武器是否都在该区域可用
            if not add_attrs_set.issubset(region_add_attrs):
                continue
            if not skills_set.issubset(region_skills):
                continue
            
            # 尝试固定附加属性
            for add_attr in add_attrs_set:
                # 计算主属性组合
                main_attrs_list = list(main_attrs_set)
                if len(main_attrs_list) <= 3:
                    # 可以直接使用所有主属性
                    score = self.calculate_group_cost(weapons_group, region_name, 'add_attr', add_attr)
                    if score < best_score:
                        best_score = score
                        best_region = region_name
                        best_fixed_type = 'add_attr'
                        best_fixed_value = add_attr
                else:
                    # 需要选择3个主属性，尝试所有组合
                    for main_combo in self.get_main_attr_combinations(main_attrs_list):
                        # 只考虑包含这些主属性的武器
                        valid_weapons = [w for w in weapons_group if w['main_attr'] in main_combo]
                        if valid_weapons:
                            score = self.calculate_group_cost(valid_weapons, region_name, 'add_attr', add_attr)
                            if score < best_score:
                                best_score = score
                                best_region = region_name
                                best_fixed_type = 'add_attr'
                                best_fixed_value = add_attr
            
            # 尝试固定技能
            for skill in skills_set:
                # 计算主属性组合
                main_attrs_list = list(main_attrs_set)
                if len(main_attrs_list) <= 3:
                    # 可以直接使用所有主属性
                    score = self.calculate_group_cost(weapons_group, region_name, 'skill', skill)
                    if score < best_score:
                        best_score = score
                        best_region = region_name
                        best_fixed_type = 'skill'
                        best_fixed_value = skill
                else:
                    # 需要选择3个主属性，尝试所有组合
                    for main_combo in self.get_main_attr_combinations(main_attrs_list):
                        # 只考虑包含这些主属性的武器
                        valid_weapons = [w for w in weapons_group if w['main_attr'] in main_combo]
                        if valid_weapons:
                            score = self.calculate_group_cost(valid_weapons, region_name, 'skill', skill)
                            if score < best_score:
                                best_score = score
                                best_region = region_name
                                best_fixed_type = 'skill'
                                best_fixed_value = skill
        
        return best_region, best_fixed_type, best_fixed_value, best_score
    
    def get_main_attr_combinations(self, main_attrs_list):
        """获取所有3主属性组合"""
        if len(main_attrs_list) <= 3:
            return [main_attrs_list]
        
        from itertools import combinations
        return list(combinations(main_attrs_list, 3))
    
    def calculate_group_cost(self, weapons_group, region_name, fixed_type, fixed_value):
        """计算武器组的刷取成本"""
        # 统计主属性分布
        main_attr_counter = Counter(w['main_attr'] for w in weapons_group)
        
        # 计算需要的主属性组合数
        if len(main_attr_counter) <= 3:
            # 一组配置即可
            # 计算最耗时的主属性组合
            max_count = max(main_attr_counter.values())
            runs_needed = math.ceil(max_count / 2)  # 每次刷取给2个基质
            
            # 每个主属性组合需要12次刷取（为了高概率获得）
            times_needed = runs_needed * 12
            
            # 应用区域权重
            cost = times_needed * self.region_cost[region_name]
            return cost
        else:
            # 需要多组配置，这里简化处理
            total_times = 0
            main_attrs_list = list(main_attr_counter.keys())
            
            # 将主属性分组，每组最多3个
            for i in range(0, len(main_attrs_list), 3):
                group_attrs = main_attrs_list[i:i+3]
                # 计算这组主属性的武器数量
                group_count = sum(main_attr_counter[attr] for attr in group_attrs)
                runs_needed = math.ceil(group_count / 2)
                total_times += runs_needed * 12
            
            # 应用区域权重
            cost = total_times * self.region_cost[region_name]
            return cost
    
    def build_attribute_tree(self, weapons):
        """构建属性树并寻找最优分组"""
        # 初始状态：所有武器未分配
        unassigned = weapons.copy()
        assignments = []
        total_cost = 0
        
        while unassigned:
            print(f"剩余未分配武器数量: {len(unassigned)}")
            
            # 统计附加属性和技能的频率
            add_attr_counter = Counter()
            skill_counter = Counter()
            
            for weapon in unassigned:
                add_attr_counter[weapon['add_attr']] += 1
                skill_counter[weapon['skill']] += 1
            
            print(f"附加属性频率: {dict(add_attr_counter)}")
            print(f"技能频率: {dict(skill_counter)}")
            
            # 找出频率最高的属性
            all_attrs = list(add_attr_counter.items()) + list(skill_counter.items())
            all_attrs.sort(key=lambda x: x[1], reverse=True)
            
            # 从最高频率开始尝试
            best_group_cost = float('inf')
            best_group_assignment = None
            best_group_weapons = None
            
            for attr, count in all_attrs:
                print(f"尝试属性: {attr} (出现{count}次)")
                
                # 根据属性类型选择武器
                if attr in add_attr_counter:
                    # 固定附加属性
                    group_weapons = [w for w in unassigned if w['add_attr'] == attr]
                    print(f"  固定附加属性 {attr}，找到 {len(group_weapons)} 个武器")
                else:
                    # 固定技能
                    group_weapons = [w for w in unassigned if w['skill'] == attr]
                    print(f"  固定技能 {attr}，找到 {len(group_weapons)} 个武器")
                
                # 尝试为这组武器找到合适的区域
                region, found_fixed_type, fixed_value, cost = self.find_region_for_group(group_weapons)
                
                if region:
                    print(f"  找到区域: {region}, 成本: {cost}")
                    if cost < best_group_cost:
                        best_group_cost = cost
                        best_group_assignment = (region, found_fixed_type, fixed_value)
                        best_group_weapons = group_weapons
            
            # 如果找到了合适的组
            if best_group_assignment and best_group_weapons:
                region, fixed_type, fixed_value = best_group_assignment
                
                print(f"\n选择了区域 {region}，固定{fixed_type}: {fixed_value}")
                print(f"包含武器: {[w['name'] for w in best_group_weapons]}")
                
                # 提取主属性组合
                main_attrs_set = set(w['main_attr'] for w in best_group_weapons)
                main_attrs_list = list(main_attrs_set)
                
                # 如果主属性超过3个，需要选择最重要的3个
                if len(main_attrs_list) > 3:
                    # 统计主属性频率
                    main_attr_counter = Counter(w['main_attr'] for w in best_group_weapons)
                    # 选择频率最高的3个
                    main_attrs_list = [attr for attr, _ in main_attr_counter.most_common(3)]
                    print(f"主属性超过3个，选择: {main_attrs_list}")
                
                # 计算刷取次数
                main_attr_counter = Counter(w['main_attr'] for w in best_group_weapons)
                # 只考虑选择的主属性
                selected_counter = {attr: main_attr_counter.get(attr, 0) for attr in main_attrs_list}
                
                # 计算需要的主属性组合
                if len(main_attrs_list) <= 3:
                    # 一组配置
                    max_count = max(selected_counter.values())
                    runs_needed = math.ceil(max_count / 2)
                    times_needed = runs_needed * 12
                else:
                    # 多组配置（这里不会发生，因为我们已经限制了最多3个）
                    times_needed = 0
                
                print(f"需要刷取次数: {times_needed}")
                
                # 创建分配记录
                if fixed_type == 'add_attr':
                    assignment = {
                        '区域': region,
                        '主属性1': main_attrs_list[0] if len(main_attrs_list) > 0 else '任意',
                        '主属性2': main_attrs_list[1] if len(main_attrs_list) > 1 else '任意',
                        '主属性3': main_attrs_list[2] if len(main_attrs_list) > 2 else '任意',
                        '附加属性': fixed_value,
                        '技能': '随机',
                        '次数': times_needed
                    }
                else:  # fixed_type == 'skill'
                    assignment = {
                        '区域': region,
                        '主属性1': main_attrs_list[0] if len(main_attrs_list) > 0 else '任意',
                        '主属性2': main_attrs_list[1] if len(main_attrs_list) > 1 else '任意',
                        '主属性3': main_attrs_list[2] if len(main_attrs_list) > 2 else '任意',
                        '附加属性': '随机',
                        '技能': fixed_value,
                        '次数': times_needed
                    }
                
                assignments.append(assignment)
                total_cost += times_needed * self.region_cost[region]
                
                # 从未分配列表中移除这些武器
                unassigned = [w for w in unassigned if w not in best_group_weapons]
                print(f"移除了 {len(best_group_weapons)} 个武器\n")
            else:
                # 如果没有找到合适的组，尝试单个武器
                if unassigned:
                    print("无法找到合适的组，尝试单个武器分配")
                    weapon = unassigned[0]
                    # 为单个武器寻找区域
                    region, fixed_type, fixed_value, cost = self.find_region_for_group([weapon])
                    
                    if region:
                        if fixed_type == 'add_attr':
                            assignment = {
                                '区域': region,
                                '主属性1': weapon['main_attr'],
                                '主属性2': '任意',
                                '主属性3': '任意',
                                '附加属性': fixed_value,
                                '技能': '随机',
                                '次数': 12  # 单个武器需要12次
                            }
                        else:
                            assignment = {
                                '区域': region,
                                '主属性1': weapon['main_attr'],
                                '主属性2': '任意',
                                '主属性3': '任意',
                                '附加属性': '随机',
                                '技能': fixed_value,
                                '次数': 12
                            }
                        
                        assignments.append(assignment)
                        total_cost += 12 * self.region_cost[region]
                        unassigned.pop(0)
                        print(f"为单个武器 {weapon['name']} 分配了区域 {region}\n")
                    else:
                        print(f"警告：无法为武器 {weapon['name']} 找到合适的区域，跳过该武器\n")
                        unassigned.pop(0)
        
        return assignments, total_cost
    
    def optimize(self):
        """执行优化"""
        print(f"开始优化 {len(self.weapons)} 个武器...")
        
        # 按武器类别分组优化（可选）
        assignments, total_cost = self.build_attribute_tree(self.weapons)
        
        return assignments, total_cost
    
    def save_results(self, assignments):
        """保存结果到CSV文件"""
        if assignments:
            df = pd.DataFrame(assignments)
            df = df[['区域', '主属性1', '主属性2', '主属性3', '附加属性', '技能', '次数']]
            df.to_csv('res.csv', index=False, encoding='utf-8-sig')
            print(f"\n结果已保存到 res.csv，共 {len(assignments)} 条策略")
        else:
            print("无结果可保存")

# 主程序
def main():
    # 创建规划器
    planner = MatrixPlannerOptimized()
    
    print(f"需要分配的武器数量: {len(planner.weapons)}")
    print(f"可用区域数量: {len(planner.regions)}")
    
    # 显示武器信息
    print("\n武器详细信息:")
    for weapon in planner.weapons:
        print(f"  {weapon['name']}: 主属性={weapon['main_attr']}, 附加属性={weapon['add_attr']}, 技能={weapon['skill']}")
    
    # 执行优化
    assignments, total_cost = planner.optimize()
    
    # 保存结果
    planner.save_results(assignments)
    
    # 打印结果
    if assignments:
        print("\n推荐刷取策略:")
        total_times = 0
        weighted_cost = 0
        
        for i, assignment in enumerate(assignments, 1):
            times = assignment['次数']
            region = assignment['区域']
            cost = times * planner.region_cost[region]
            
            total_times += times
            weighted_cost += cost
            
            print(f"{i}. {assignment['区域']}, {assignment['主属性1']}, {assignment['主属性2']}, {assignment['主属性3']}, {assignment['附加属性']}, {assignment['技能']}, {times}")
        
        print(f"\n统计信息:")
        print(f"总刷取次数: {total_times}")
        print(f"加权总成本: {weighted_cost}")
        print(f"策略数量: {len(assignments)}")

if __name__ == "__main__":
    main()