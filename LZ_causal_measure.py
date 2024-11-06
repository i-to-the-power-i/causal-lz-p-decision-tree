import numpy as np


def LZ_penalty(sequence_gr, sequence_cmp):
    sub_strings_gr = set()
    sub_strings_cmp = set()
    k_cap = set()
    ind_gr = 0
    inc_gr = 1
    ind_cmp = 0
    inc_cmp = 1
    #find grammar of x
    while True:
        #if dictionary of x has to be built
        if ind_gr + inc_gr <= len(sequence_gr):
            sub_str_gr = sequence_gr[ind_gr : ind_gr + inc_gr]
            if sub_str_gr in sub_strings_gr:
                inc_gr += 1
            else:
                sub_strings_gr.add(sub_str_gr)
                ind_gr += inc_gr
                inc_gr = 1

                while True:
                    sub_str_cmp = sequence_cmp[ind_cmp : ind_cmp + inc_cmp]
                    if ind_cmp + inc_cmp > len(sequence_cmp):
                        break
                    if sub_str_cmp in sub_strings_gr and sub_str_cmp not in sub_strings_cmp:
                        sub_strings_cmp.add(sub_str_cmp)
                        k_cap.add(sub_str_cmp)
                        ind_cmp += inc_cmp
                        inc_cmp = 1
                        break
                    if sub_str_cmp in sub_strings_cmp:
                        inc_cmp += 1
                    else:
                        sub_strings_cmp.add(sub_str_cmp)
                        ind_cmp += inc_cmp
                        inc_cmp = 1
                        break
                    # print(sub_strings_gr)
                    # print(sub_strings_cmp)
                    # print(k_cap)
                    # print('\nmeow')
 
                if ind_cmp + inc_cmp > len(sequence_cmp):
                        break
                
        else:
            while True:
                sub_str_cmp = sequence_cmp[ind_cmp : ind_cmp+inc_cmp]
                if ind_cmp + inc_cmp > len(sequence_cmp):
                    break
                if sub_str_cmp in sub_strings_gr and sub_str_cmp not in sub_strings_cmp:
                    sub_strings_cmp.add(sub_str_cmp)
                    k_cap.add(sub_str_cmp)
                    ind_cmp += inc_cmp
                    inc_cmp = 1
                if sub_str_cmp in sub_strings_cmp:
                    inc_cmp += 1
                else:
                    sub_strings_cmp.add(sub_str_cmp)
                    ind_cmp += inc_cmp
                    inc_cmp = 1
                # print(sub_strings_gr)
                # print(sub_strings_cmp)
                # print(k_cap)
                # print('\nhi')
            if ind_cmp + inc_cmp > len(sequence_cmp):
                break
    # print(sub_strings_gr)
    # print(sub_strings_cmp)
    # print(k_cap)
    # print('\nhi')

    return len(sub_strings_cmp) - len(k_cap)
   
def calc_penalty(x,y):
    return LZ_penalty(x,y), LZ_penalty(y,x)