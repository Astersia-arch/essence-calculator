#!/usr/bin/env python3
"""
终末地武器基质刷取规划器 - GUI版本
"""

import pandas as pd
from typing import Set, Dict, List, Tuple, Optional
from dataclasses import dataclass
from collections import Counter
import logging
from pathlib import Path
from enum import Enum
import tkinter as tk
from tkinter import ttk, messagebox
import math

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


class AttributeType(Enum):
    SUB = "sub"
    SKILL = "skill"


@dataclass(frozen=True)
class Weapon:
    name: str
    weapon_type: str
    rarity: int
    main_attrs: Tuple[str, ...]
    sub_attr: str
    skill: str
    raw_data: Tuple  # 原始行数据，用于动态显示
    
    def __hash__(self):
        return hash((self.name, self.weapon_type, self.rarity))


@dataclass
class Zone:
    name: str
    sub_attrs: Set[str]
    skills: Set[str]
    cost_multiplier: float = 1.0


@dataclass
class Strategy:
    zone: str
    main_attrs: Tuple[str, ...]
    sub_attr: Optional[str]
    skill: Optional[str]
    weapons: List[Weapon]
    target_count: int
    
    def __post_init__(self):
        if self.sub_attr is not None and self.skill is not None:
            raise ValueError("只能固定一种属性")
    
    @property
    def fixed_type(self) -> AttributeType:
        return AttributeType.SUB if self.sub_attr else AttributeType.SKILL


class EssenceCalculator:
    def __init__(self, data_dir: str = "."):
        self.data_dir = Path(data_dir)
        self.weapons: Dict[str, Weapon] = {}
        self.zones: Dict[str, Zone] = {}
        self.columns: List[str] = []  # 动态表头
        
    def load_data(self) -> None:
        try:
            df_weapons = pd.read_csv(self.data_dir / "weapons.csv", encoding='utf-8')
            self.columns = df_weapons.columns.tolist()  # 保存表头
            
            df_weapons = df_weapons.dropna(subset=['名称', '主属性', '附加属性', '技能'])
            
            for _, row in df_weapons.iterrows():
                weapon = Weapon(
                    name=str(row['名称']),
                    weapon_type=str(row.get('类别', '未知')),
                    rarity=int(row.get('稀有度', 5)),
                    main_attrs=tuple(str(row['主属性']).split('/')),
                    sub_attr=str(row['附加属性']),
                    skill=str(row['技能']),
                    raw_data=tuple(row[col] for col in self.columns)  # 保存原始行
                )
                self.weapons[weapon.name] = weapon
                
            logger.info(f"已加载 {len(self.weapons)} 个武器，列: {self.columns}")
            
            df_zones = pd.read_csv(self.data_dir / "area.csv", encoding='utf-8')
            
            for _, row in df_zones.iterrows():
                zone_name = str(row['区域'])
                sub_attrs = {str(row[f'附加属性{i}']) for i in range(1, 9) 
                           if f'附加属性{i}' in row and pd.notna(row[f'附加属性{i}'])}
                skills = {str(row[f'技能{i}']) for i in range(1, 9) 
                         if f'技能{i}' in row and pd.notna(row[f'技能{i}'])}
                
                cost = float(row.get('成本', 1.0)) if '成本' in row else 1.0
                
                self.zones[zone_name] = Zone(
                    name=zone_name,
                    sub_attrs=sub_attrs,
                    skills=skills,
                    cost_multiplier=cost
                )
                
            logger.info(f"已加载 {len(self.zones)} 个区域")
            
        except FileNotFoundError as e:
            logger.error(f"文件未找到: {e}")
            raise
        except Exception as e:
            logger.error(f"数据加载失败: {e}")
            raise

    def analyze_optimal_strategy(self, weapons: List[Weapon]) -> List[Strategy]:
        remaining = set(weapons)
        strategies = []
        
        while remaining:
            best = self._find_best_strategy(remaining)
            if not best:
                break
            
            strategies.append(best)
            remaining -= set(best.weapons)
            logger.info(f"分组: {best.zone} | {len(best.weapons)}个武器 | 剩余{len(remaining)}个")
        
        return strategies

    def _find_best_strategy(self, weapons_set: Set[Weapon]) -> Optional[Strategy]:
        if not weapons_set:
            return None

        weapons_list = list(weapons_set)
        
        sub_freq = Counter()
        skill_freq = Counter()
        
        for w in weapons_list:
            weight = w.rarity / 5.0
            sub_freq[w.sub_attr] += weight
            skill_freq[w.skill] += weight
        
        best_score = float('-inf')
        best_strategy = None
        
        for fixed_type, freq_dict in [(AttributeType.SUB, sub_freq), 
                                      (AttributeType.SKILL, skill_freq)]:
            
            for attr_name, freq in freq_dict.most_common():
                if fixed_type == AttributeType.SUB:
                    matched = [w for w in weapons_list if w.sub_attr == attr_name]
                else:
                    matched = [w for w in weapons_list if w.skill == attr_name]
                
                if not matched:
                    continue
                
                zone_name, zone = self._find_best_zone(matched, fixed_type, attr_name)
                if not zone:
                    continue
                
                main_attrs = self._select_optimal_main_attrs(matched)
                
                coverage = len(matched)
                cost_factor = 1 / zone.cost_multiplier
                variety = len(set(w.name for w in matched)) / coverage
                score = coverage * cost_factor * variety * (1 + freq * 0.1)
                
                if score > best_score:
                    best_score = score
                    runs = self._calculate_runs(len(matched), len(main_attrs))
                    
                    best_strategy = Strategy(
                        zone=zone_name,
                        main_attrs=main_attrs,
                        sub_attr=attr_name if fixed_type == AttributeType.SUB else None,
                        skill=attr_name if fixed_type == AttributeType.SKILL else None,
                        weapons=matched,
                        target_count=int(runs)
                    )
        
        return best_strategy

    def _find_best_zone(self, weapons: List[Weapon], 
                       fixed_type: AttributeType, 
                       fixed_attr: str) -> Tuple[str, Optional[Zone]]:
        candidates = []
        
        for zone_name, zone in self.zones.items():
            if fixed_type == AttributeType.SUB:
                if fixed_attr not in zone.sub_attrs:
                    continue
            else:
                if fixed_attr not in zone.skills:
                    continue
            
            compatible = 0
            for w in weapons:
                if fixed_type == AttributeType.SUB:
                    if w.skill in zone.skills:
                        compatible += 1
                else:
                    if w.sub_attr in zone.sub_attrs:
                        compatible += 1
            
            score = compatible / zone.cost_multiplier
            candidates.append((zone_name, zone, score))
        
        if not candidates:
            return "未知区域", None
            
        candidates.sort(key=lambda x: x[2], reverse=True)
        return candidates[0][0], candidates[0][1]

    def _select_optimal_main_attrs(self, weapons: List[Weapon]) -> Tuple[str, ...]:
        all_attrs = []
        for w in weapons:
            all_attrs.extend(w.main_attrs)
        
        top3 = [attr for attr, _ in Counter(all_attrs).most_common(3)]
        return tuple(top3)

    def _calculate_runs(self, weapon_count: int, main_attr_count: int) -> int:
        drops_per_run = 2
        
        if main_attr_count <= 3:
            combos = 1
        else:
            combos = math.comb(main_attr_count, 3)
        
        runs = (weapon_count / drops_per_run) * 1.5 * combos
        return max(1, int(runs))


class EssenceCalculatorGUI:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("终末地武器基质刷取规划器")
        self.root.geometry("950x850")
        
        self.calculator = EssenceCalculator()
        try:
            self.calculator.load_data()
        except Exception as e:
            messagebox.showerror("错误", f"加载数据失败: {e}")
            raise
        
        self.selected_weapons: Dict[str, Weapon] = {}  # 改用dict便于删除
        
        self._build_ui()
        self._load_weapons_to_source()
        
    def _calc_width(self, text: str, font_size: int = 10) -> int:
        return len(str(text)) * 2 * font_size + 16
    
    def _create_frame(self, parent, title: str, row: int, col: int, 
                     rowspan: int = 1, colspan: int = 1) -> ttk.LabelFrame:
        frame = ttk.LabelFrame(parent, text=title, padding="5")
        frame.grid(row=row, column=col, rowspan=rowspan, columnspan=colspan,
                  sticky=(tk.W, tk.E, tk.N, tk.S), padx=5, pady=5)
        frame.columnconfigure(0, weight=1)
        frame.rowconfigure(0, weight=1)
        return frame
    
    def _create_tree(self, parent, columns: List[str], widths: Dict[str, int], 
                    row: int, height: int = 15) -> ttk.Treeview:
        tree = ttk.Treeview(parent, columns=columns, show="headings", 
                           selectmode="extended", height=height)
        tree.grid(row=row, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        for col in columns:
            tree.heading(col, text=col)
            w = widths.get(col, 100)
            tree.column(col, width=w, anchor='center', minwidth=50)
        
        vsb = ttk.Scrollbar(parent, orient=tk.VERTICAL, command=tree.yview)
        vsb.grid(row=row, column=1, sticky=(tk.N, tk.S))
        tree.configure(yscrollcommand=vsb.set)
        
        return tree
    
    def _create_button_group(self, parent, commands: List[Tuple[str, callable]], row: int):
        frame = ttk.Frame(parent)
        frame.grid(row=row, column=0, pady=5)
        for text, cmd in commands:
            ttk.Button(frame, text=text, command=cmd).pack(side=tk.LEFT, padx=2)
    
    def _build_ui(self):
        main = ttk.Frame(self.root, padding="10")
        main.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main.columnconfigure(0, weight=1)
        main.columnconfigure(1, weight=1)
        main.rowconfigure(0, weight=3)
        main.rowconfigure(1, weight=2)
        
        # 获取所有列名
        all_cols = self.calculator.columns
        
        # 左侧显示所有列
        left = self._create_frame(main, "武器库", 0, 0)
        
        # 筛选栏
        filter_bar = ttk.Frame(left)
        filter_bar.grid(row=0, column=0, sticky=(tk.W, tk.E), pady=5)
        
        # 动态创建筛选（使用前3列作为筛选选项）
        self.filter_vars = []
        for i, col_name in enumerate(all_cols[:3]):  # 前3列通常是类别、稀有度、名称
            ttk.Label(filter_bar, text=f"{col_name}:").pack(side=tk.LEFT, padx=2)
            var = tk.StringVar(value="全部")
            self.filter_vars.append((col_name, var))
            values = ["全部"] + sorted(list(set(str(w.raw_data[i]) for w in self.calculator.weapons.values())))
            cb = ttk.Combobox(filter_bar, textvariable=var, values=values, state="readonly", width=10)
            cb.pack(side=tk.LEFT, padx=2)
            var.trace('w', lambda *args: self._apply_filter())
        
        ttk.Label(filter_bar, text="搜索:").pack(side=tk.LEFT, padx=2)
        self.search_var = tk.StringVar()
        ttk.Entry(filter_bar, textvariable=self.search_var, width=15).pack(side=tk.LEFT, padx=2)
        self.search_var.trace('w', lambda *args: self._apply_filter())
        
        # 左侧表格 - 显示所有列
        widths = {}
        for i, col in enumerate(all_cols):
            max_len = max(len(col), max(len(str(w.raw_data[i])) for w in self.calculator.weapons.values()))
            widths[col] = self._calc_width(max_len)
        
        self.source_tree = self._create_tree(left, all_cols, widths, 1)
        
        # 绑定双击添加事件
        self.source_tree.bind('<Double-1>', lambda e: self._add_selected())
        
        self._create_button_group(left, [("➡ 添加选中", self._add_selected)], 2)
        
        # 右侧 - 只显示前3列（通常是类别、稀有度、名称）
        right = self._create_frame(main, "已选武器", 0, 1)
        right_cols = all_cols[:3]  # 取前3列
        right_widths = {col: widths[col] for col in right_cols}
        
        self.target_tree = self._create_tree(right, right_cols, right_widths, 0)
        self.target_tree.bind('<Delete>', lambda e: self._remove_selected())  # 支持Delete键删除
        
        self._create_button_group(right, [
            ("⬅ 删除选中", self._remove_selected),
            ("⚠ 清空全部", self._clear_all)
        ], 1)
        
        # 底部 - 策略结果
        bottom = self._create_frame(main, "刷取策略", 1, 0, colspan=2)
        
        res_cols = ['区域', '主属性1', '主属性2', '主属性3', '附加属性', '技能', '次数', '武器列表']
        res_widths = {
            '区域': 80, '主属性1': 60, '主属性2': 60, '主属性3': 60,
            '附加属性': 70, '技能': 70, '次数': 50, '武器列表': 600
        }
        
        self.result_tree = self._create_tree(bottom, res_cols, res_widths, 0)
        
        # 水平滚动条给武器列表
        hsb = ttk.Scrollbar(bottom, orient=tk.HORIZONTAL, command=self.result_tree.xview)
        hsb.grid(row=1, column=0, sticky=(tk.W, tk.E))
        self.result_tree.configure(xscrollcommand=hsb.set)
        
        self._create_button_group(bottom, [("🚀 开始计算策略", self._calculate)], 2)
    
    def _load_weapons_to_source(self):
        """加载武器到左侧表格"""
        all_cols = self.calculator.columns
        for weapon in self.calculator.weapons.values():
            self.source_tree.insert('', tk.END, values=weapon.raw_data, iid=weapon.name)
    
    def _apply_filter(self):
        """应用筛选"""
        all_cols = self.calculator.columns
        
        # 获取筛选条件（列名->值）
        filters = {}
        for col_name, var in self.filter_vars:
            val = var.get()
            if val != "全部":
                filters[col_name] = val
        
        search = self.search_var.get().lower()
        
        self.source_tree.delete(*self.source_tree.get_children())
        
        for weapon in self.calculator.weapons.values():
            # 检查固定筛选项
            match = True
            for col_name, filter_val in filters.items():
                col_idx = all_cols.index(col_name)
                if str(weapon.raw_data[col_idx]) != filter_val:
                    match = False
                    break
            
            if not match:
                continue
            
            # 检查搜索（在所有字段中搜索）
            if search and search not in ' '.join(str(x) for x in weapon.raw_data).lower():
                continue
            
            self.source_tree.insert('', tk.END, values=weapon.raw_data, iid=weapon.name)
    
    def _add_selected(self):
        """添加选中到右侧"""
        selection = self.source_tree.selection()
        if not selection:
            messagebox.showinfo("提示", "请先在左侧选择武器")
            return
        
        added = 0
        for item in selection:
            if item not in self.selected_weapons:
                weapon = self.calculator.weapons[item]
                self.selected_weapons[item] = weapon
                # 右侧只显示前3列
                self.target_tree.insert('', tk.END, values=weapon.raw_data[:3], iid=item)
                added += 1
        
        if added == 0:
            messagebox.showinfo("提示", "选中的武器已在列表中")
    
    def _remove_selected(self):
        """从右侧删除选中"""
        selection = self.target_tree.selection()
        if not selection:
            messagebox.showinfo("提示", "请先在右侧选择要删除的武器")
            return
        
        count = len(selection)
        if count > 1:
            if not messagebox.askyesno("确认", f"确定删除选中的 {count} 个武器吗？"):
                return
        
        for item in selection:
            self.target_tree.delete(item)
            del self.selected_weapons[item]
    
    def _clear_all(self):
        """清空全部"""
        if not self.selected_weapons:
            return
        
        if messagebox.askyesno("确认", f"确定清空全部 {len(self.selected_weapons)} 个武器吗？"):
            self.target_tree.delete(*self.target_tree.get_children())
            self.selected_weapons.clear()
    
    def _calculate(self):
        """计算策略"""
        if not self.selected_weapons:
            messagebox.showwarning("警告", "请先选择武器")
            return
        
        self.result_tree.delete(*self.result_tree.get_children())
        
        try:
            strategies = self.calculator.analyze_optimal_strategy(list(self.selected_weapons.values()))
            
            if not strategies:
                messagebox.showinfo("提示", "无法生成有效策略")
                return
            
            for s in strategies:
                weapon_names = '; '.join([w.name for w in s.weapons])
                vals = (
                    s.zone,
                    s.main_attrs[0] if len(s.main_attrs) > 0 else '',
                    s.main_attrs[1] if len(s.main_attrs) > 1 else '',
                    s.main_attrs[2] if len(s.main_attrs) > 2 else '',
                    s.sub_attr if s.sub_attr else '随机',
                    s.skill if s.skill else '随机',
                    s.target_count,
                    weapon_names
                )
                self.result_tree.insert('', tk.END, values=vals)
            
            self._export_to_csv(strategies)
            messagebox.showinfo("完成", f"生成 {len(strategies)} 条策略，已保存至 res.csv")
            
        except Exception as e:
            messagebox.showerror("错误", f"计算失败: {e}")
    
    def _export_to_csv(self, strategies: List[Strategy]):
        data = []
        for s in strategies:
            data.append({
                '区域': s.zone,
                '主属性1': s.main_attrs[0] if len(s.main_attrs) > 0 else '',
                '主属性2': s.main_attrs[1] if len(s.main_attrs) > 1 else '',
                '主属性3': s.main_attrs[2] if len(s.main_attrs) > 2 else '',
                '附加属性': s.sub_attr if s.sub_attr else '随机',
                '技能': s.skill if s.skill else '随机',
                '次数': s.target_count,
                '武器列表': ';'.join([w.name for w in s.weapons])
            })
        
        pd.DataFrame(data).to_csv('res.csv', index=False, encoding='utf-8-sig')


def main():
    root = tk.Tk()
    app = EssenceCalculatorGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
