import numpy as np

def LZ_penalty(sequence_gr, sequence_cmp):
    #Initialize Grammar Sets G_x and G_y
    sub_strings_gr = set()
    sub_strings_cmp = set()
    #Initialize SubString pointers
    ind_gr = 0
    inc_gr = 1
    ind_cmp = 0
    inc_cmp = 1
    #Initialize overlap extent to 0
    overlap = 0
    #Main loop
    while True:
        #Find a substring not in G_x
        while True:
            if ind_gr + inc_gr > len(sequence_gr):
                break
            sub_str_gr = sequence_gr[ind_gr : ind_gr + inc_gr]
            if sub_str_gr in sub_strings_gr:
                inc_gr += 1
            else:
                break
        #Add it to G_x if such a substring exists. Move pointers to next part of the string.
        if ind_gr + inc_gr <= len(sequence_gr):
            sub_strings_gr.add(sub_str_gr)
            ind_gr += inc_gr
            inc_gr = 1
        
        #Do the same for sequence y.
        while True:
            if ind_cmp + inc_cmp > len(sequence_cmp): 
                break
            sub_str_cmp = sequence_cmp[ind_cmp : ind_cmp + inc_cmp]
            if sub_str_cmp in sub_strings_cmp:
                inc_cmp += 1
            else: 
                break
        if ind_cmp + inc_cmp > len(sequence_cmp):
            break
        sub_strings_cmp.add(sub_str_cmp)
        #If this substring is already present in G_x, increase overlap by one.
        if sub_str_cmp in sub_strings_gr:
            overlap += 1
        ind_cmp += inc_cmp
        inc_cmp = 1

        # print(sub_strings_gr, sub_strings_cmp)
    return len(sub_strings_cmp) - overlap
   
def calc_penalty(x,y):
    return LZ_penalty(x,y), LZ_penalty(y,x)
print(LZ_penalty("100","100"))