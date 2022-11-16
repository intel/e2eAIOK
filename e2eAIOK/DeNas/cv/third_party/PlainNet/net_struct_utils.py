def _get_right_parentheses_index_(s):
    # assert s[0] == '('
    left_paren_count = 0
    for index, x in enumerate(s):

        if x == '(':
            left_paren_count += 1
        elif x == ')':
            left_paren_count -= 1
            if left_paren_count == 0:
                return index
        else:
            pass
    return None

def pretty_format(plainnet_str, indent=2):
    the_formated_str = ''
    indent_str = ''
    if indent >= 1:
        indent_str = ''.join(['  '] * indent)

    # print(indent_str, end='')
    the_formated_str += indent_str

    s = plainnet_str
    while len(s) > 0:
        if s[0] == ';':
            # print(';\n' + indent_str, end='')
            the_formated_str += ';\n' + indent_str
            s = s[1:]

        left_par_idx = s.find('(')
        assert left_par_idx is not None
        right_par_idx = _get_right_parentheses_index_(s)
        the_block_class_name = s[0:left_par_idx]

        if the_block_class_name in ['MultiSumBlock', 'MultiCatBlock','MultiGroupBlock']:
            # print('\n' + indent_str + the_block_class_name + '(')
            sub_str = s[left_par_idx + 1:right_par_idx]

            # find block_name
            tmp_idx = sub_str.find('|')
            if tmp_idx < 0:
                tmp_block_name = 'no_name'
            else:
                tmp_block_name = sub_str[0:tmp_idx]
                sub_str = sub_str[tmp_idx+1:]

            if len(tmp_block_name) > 8:
                tmp_block_name = tmp_block_name[0:4] + tmp_block_name[-4:]

            the_formated_str += '\n' + indent_str + the_block_class_name + '({}|\n'.format(tmp_block_name)

            the_formated_str += pretty_format(sub_str, indent + 1)
            # print('\n' + indent_str + ')')
            # print(indent_str, end='')
            the_formated_str += '\n' + indent_str + ')\n' + indent_str
        elif the_block_class_name in ['ResBlock']:
            # print('\n' + indent_str + the_block_class_name + '(')
            in_channels = None
            the_stride = None
            sub_str = s[left_par_idx + 1:right_par_idx]
            # find block_name
            tmp_idx = sub_str.find('|')
            if tmp_idx < 0:
                tmp_block_name = 'no_name'
            else:
                tmp_block_name = sub_str[0:tmp_idx]
                sub_str = sub_str[tmp_idx + 1:]

            first_comma_index = sub_str.find(',')
            if first_comma_index < 0 or not sub_str[0:first_comma_index].isdigit():
                in_channels = None
            else:
                in_channels = int(sub_str[0:first_comma_index])
                sub_str = sub_str[first_comma_index+1:]
                second_comma_index = sub_str.find(',')
                if second_comma_index < 0 or not sub_str[0:second_comma_index].isdigit():
                    the_stride = None
                else:
                    the_stride = int(sub_str[0:second_comma_index])
                    sub_str = sub_str[second_comma_index + 1:]
                pass
            pass

            if len(tmp_block_name) > 8:
                tmp_block_name = tmp_block_name[0:4] + tmp_block_name[-4:]

            the_formated_str += '\n' + indent_str + the_block_class_name + '({}|'.format(tmp_block_name)
            if in_channels is not None:
                the_formated_str += '{},'.format(in_channels)
            else:
                the_formated_str += ','

            if the_stride is not None:
                the_formated_str += '{},'.format(the_stride)
            else:
                the_formated_str += ','

            the_formated_str += '\n'

            the_formated_str += pretty_format(sub_str, indent + 1)
            # print('\n' + indent_str + ')')
            # print(indent_str, end='')
            the_formated_str += '\n' + indent_str + ')\n' + indent_str
        else:
            # print(s[0:right_par_idx+1], end='')
            sub_str = s[left_par_idx + 1:right_par_idx]
            # find block_name
            tmp_idx = sub_str.find('|')
            if tmp_idx < 0:
                tmp_block_name = 'no_name'
            else:
                tmp_block_name = sub_str[0:tmp_idx]
                sub_str = sub_str[tmp_idx + 1:]

            if len(tmp_block_name) > 8:
                tmp_block_name = tmp_block_name[0:4] + tmp_block_name[-4:]

            the_formated_str += the_block_class_name + '({}|'.format(tmp_block_name) + sub_str + ')'

        s = s[right_par_idx+1:]
    pass  # end while

    return the_formated_str

def _create_netblock_list_from_str_(s, all_netblocks_dict, no_create=False, **kwargs):
    block_list = []
    while len(s) > 0:
        is_found_block_class = False
        for the_block_class_name in all_netblocks_dict.keys():
            tmp_idx = s.find('(')
            if tmp_idx > 0 and s[0:tmp_idx] == the_block_class_name:
                is_found_block_class = True
                the_block_class = all_netblocks_dict[the_block_class_name]
                the_block, remaining_s = the_block_class.create_from_str(s, no_create=no_create, **kwargs)
                if the_block is not None:
                    block_list.append(the_block)
                s = remaining_s
                if len(s) > 0 and s[0] == ';':
                    return block_list, s[1:]
                break
            pass  # end if
        pass  # end for
        assert is_found_block_class
    pass  # end while
    return block_list, ''

def create_netblock_list_from_str(s, all_netblocks_dict, no_create=False, **kwargs):
    the_list, remaining_s = _create_netblock_list_from_str_(s, all_netblocks_dict, no_create=no_create, **kwargs)
    assert len(remaining_s) == 0
    return the_list

def add_SE_block(structure_str: str):
    new_str = ''
    RELU = 'RELU'
    offset = 4

    idx = structure_str.find(RELU)
    while idx >= 0:
        new_str += structure_str[0: idx]
        structure_str = structure_str[idx:]
        r_idx = _get_right_parentheses_index_(structure_str[offset:]) + offset
        channels = structure_str[offset + 1:r_idx]
        new_str += 'RELU({})SE({})'.format(channels, channels)
        structure_str = structure_str[r_idx + 1:]
        idx = structure_str.find(RELU)
    pass

    new_str += structure_str
    return new_str
