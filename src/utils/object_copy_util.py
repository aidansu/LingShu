"""
对象属性复制工具类
类似于Spring BeanUtils.copyProperties的功能
"""
from typing import Any, Set, Dict
import logging

logger = logging.getLogger(__name__)


class ObjectCopyUtil:
    """对象属性复制工具类"""
    
    @staticmethod
    def copy_properties(target: Any, source: Any, ignore_properties: Set[str] | None = None) -> None:
        """
        复制源对象的属性到目标对象
        
        Args:
            target: 目标对象
            source: 源对象
            ignore_properties: 要忽略的属性名集合
        """
        if not source or target is None:
            return
        
        ignore_set = ignore_properties or set()
        
        # 获取源对象的所有属性
        source_dict = source.__dict__ if hasattr(source, '__dict__') else {}
        
        for attr_name, attr_value in source_dict.items():
            # 跳过忽略的属性
            if attr_name in ignore_set:
                continue
                
            # 跳过私有属性
            if attr_name.startswith('_'):
                continue
                
            # 如果设置忽略None值且当前值为None，则跳过
            if attr_value is None:
                continue
                
            # 检查目标对象是否有该属性
            if hasattr(target, attr_name):
                try:
                    setattr(target, attr_name, attr_value)
                except AttributeError as e:
                    logger.warning(f"无法设置属性 {attr_name}: {e}")
    
    @staticmethod
    def copy_properties_with_mapping(target: Any, source: Any, property_mapping: Dict[str, str]) -> None:
        """
        使用属性映射复制对象属性
        
        Args:
            target: 目标对象
            source: 源对象  
            property_mapping: 属性映射字典 {源属性名: 目标属性名}
        """
        if not source or not target:
            return
            
        source_dict = source.__dict__ if hasattr(source, '__dict__') else {}
        
        for source_attr, target_attr in property_mapping.items():
            if source_attr in source_dict:
                try:
                    setattr(target, target_attr, source_dict[source_attr])
                except AttributeError as e:
                    logger.warning(f"无法设置属性 {target_attr}: {e}")
    
    @staticmethod
    def to_dict(obj: Any, exclude_none: bool = False, exclude_private: bool = True) -> Dict[str, Any]:
        """
        将对象转换为字典
        
        Args:
            obj: 要转换的对象
            exclude_none: 是否排除None值
            exclude_private: 是否排除私有属性
            
        Returns:
            对象属性字典
        """
        if not hasattr(obj, '__dict__'):
            return {}
            
        result = {}
        for key, value in obj.__dict__.items():
            # 排除私有属性
            if exclude_private and key.startswith('_'):
                continue
                
            # 排除None值
            if exclude_none and value is None:
                continue
                
            result[key] = value
            
        return result
    
    @staticmethod  
    def from_dict(cls: type, data: Dict[str, Any]) -> Any | None:
        """
        从字典创建对象实例
        
        Args:
            cls: 目标类
            data: 属性数据字典
            
        Returns:
            创建的对象实例
        """
        try:
            # 如果类有__init__方法且接受关键字参数
            instance = cls()
            for key, value in data.items():
                if hasattr(instance, key):
                    setattr(instance, key, value)
            return instance
        except Exception as e:
            logger.error(f"从字典创建对象失败: {e}")
            return None 