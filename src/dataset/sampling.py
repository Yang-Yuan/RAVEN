# -*- coding: utf-8 -*-


import numpy as np
from scipy.misc import comb

from const import MAX_COMPONENTS, RULE_ATTR
from Rule import Rule_Wrapper


def sample_rules():
    """First sample # components; for each component, sample a rule on each attribute.
    """
    num_components = np.random.randint(1, MAX_COMPONENTS + 1)
    all_rules = []
    for i in range(num_components):
        all_rules_component = []
        for j in range(len(RULE_ATTR)):
            idx = np.random.choice(len(RULE_ATTR[j]))
            name_attr_param = RULE_ATTR[j][idx]
            all_rules_component.append(Rule_Wrapper(name_attr_param[0], name_attr_param[1], name_attr_param[2], component_idx=i))
        all_rules.append(all_rules_component)
    return all_rules

# pay attention to Position Arithmetic, new entities (resample)
def sample_attr_avail(rule_groups, row_3_3):
    """Sample available attributes whose values could be modified.
    Arguments:
        rule_groups(list of list of Rule): a list of rules to apply to the component
        row_3_3(AoTNode): the answer AoT
    Returns:
        ret(list of list): [component_idx, attr, available_times, constraints]
    """
    ret = []
    for i in range(len(rule_groups)):

        # this is not start_node_layout, but under the current implementation
        # they should be the same
        start_node_layout = row_3_3.children[0].children[i].children[0]
        most_num = len(start_node_layout.position.values)

        row_3_3_layout = row_3_3.children[0].children[i].children[0]
        uni = row_3_3_layout.uniformity.get_value()
        num = row_3_3_layout.number.get_value()

        rule_group = rule_groups[i]
        num_pos_rule = rule_group[0]

        # Number/Position
        # If Rule on Number: Only change Number
        # If Rule on Position: Both Number and Position could be changed
        if num_pos_rule.attr == "Number":
            num_times = 0
            min_level = start_node_layout.orig_layout_constraint["Number"][0]
            max_level = start_node_layout.orig_layout_constraint["Number"][1]
            for k in range(min_level, max_level + 1):
                if k + 1 != num:
                    num_times += comb(most_num, k + 1)  # for a number of entities, they can be placed at different positions
            if num_times > 0:
                ret.append([i, "Number", num_times, min_level, max_level])
        # Constant or on Position
        else:
            num_times = 0
            min_level = start_node_layout.orig_layout_constraint["Number"][0]
            max_level = start_node_layout.orig_layout_constraint["Number"][1]
            for k in range(min_level, max_level + 1):
                if k + 1 != num:
                    num_times += comb(most_num, k + 1)
            if num_times > 0:
                ret.append([i, "Number", num_times, min_level, max_level])

            pos_times = comb(most_num, num)
            pos_times -= 1
            if pos_times > 0:
                ret.append([i, "Position", pos_times, None, None])

        # Type, Size, Color
        for j in range(1, len(rule_group)):
            rule = rule_group[j]
            rule_attr = rule.attr
            min_level = start_node_layout.orig_entity_constraint[rule_attr][0]
            max_level = start_node_layout.orig_entity_constraint[rule_attr][1] 
            if rule.name == "Constant":

                # Given the current implementation of creating start_node and applying rules incrementally,
                # the constant rules on type, size, and color will not be effective in all cases.
                # In particular, the constant rule will take effect, when
                # (1) uni == True.
                #     When uni==False and num_pos_rule is not constant, this rule will resample each entity in each panel individually,
                #     thus being different. Then, when it is applying the constant rules on type, size, and color, this rule simply
                #     returns the input panel.
                # (2) When uni == False, num_pos_rule has to be constant for constant rules on type, size, and color to work.
                #     This is also because of the current implementation of constant rule.
                # (3) When both the above two points are not satisfied, the progression and distribute_three on position will not resample entities.
                if uni or num_pos_rule.name == "Constant" or \
                        (rule_group[0].attr == "Position" and
                         (rule_group[0].name == "Progression" or rule_group[0].name == "Distribute_Three")):
                    times = max_level - min_level + 1
                    times = times - 1
                    if times > 0:
                        ret.append([i, rule_attr, times, min_level, max_level])
            else:
                times = max_level - min_level + 1
                times = times - 1
                if times > 0:
                    ret.append([i, rule_attr, times, min_level, max_level])
    return ret


def sample_attr(attrs_list):
    """Given the attr_avail list, sample one attribute to modify the value.
    If the available times becomes zero, delete it.
    Arguments:
        attrs_list(list of list): a flat component of available attributes 
            to change the values; consisting of different component indexes
    """
    attr_idx = np.random.choice(len(attrs_list))
    component_idx, attr_name, _, min_level, max_level = attrs_list[attr_idx]
    attrs_list[attr_idx][2] -= 1
    if attrs_list[attr_idx][2] == 0:
        del attrs_list[attr_idx]
    return component_idx, attr_name, min_level, max_level
