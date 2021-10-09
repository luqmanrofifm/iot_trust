import pandas as pd
import numpy as np
import datetime as dt
import random
from itertools import permutations
import time

# number_of_node merupakan jumlah node pada komunitas, pada nilai tersebut nilai minimalnya adalah 2
# comm_id merupakan ID yang digunakan ketika membuat objek, default valuenya adalah A

# prob_connection merupakan persentasi jumlah relasi pada setiap node dan berbentuk array
# sebagai contoh misalkan terdapat 3 node, maka prob_connection nya dapat diisi menjadi [0.1, 0.2, 0.3]
# hal tersebut memiliki arti node 1 memiliki jumlah relasi 0.1 * total relasi yang memungkinkan
# pada contoh ini relasi yang memungkinkan pada setiap node adalah 2 (relasi ke diri sendiri tidak termasuk)
# default value dari prob_connection adalah 0, jadi jika prob_connection tidak diisi dengan array,
# maka persentase relasi tiap node akan dirandom

class Community:
  def __init__ (self,number_of_node, comm_id='A', prob_connection=0):
    self.comm_id = comm_id

    if number_of_node > 1:
      self.number_of_node = number_of_node
    else :
      raise ValueError('Jumlah node minimal adalah 2')

    self.prob_connection = prob_connection
    self.matrix_relation = 0
    self.table_of_node = Community.generate_table_node(self, self.number_of_node)
    self.table_of_relation = Community.generate_table_relation(self)

  # fungsi untuk generate tabel yang berisi daftar node pada suatu komunitas
  def generate_table_node(self, number_node):
    comm_id = self.comm_id
    df = pd.DataFrame(columns=['Comm_ID','Node_ID','X','Y','Trust','RAM'])
    for i in range(number_node):
      df = df.append({'Comm_ID':comm_id, 'Node_ID':comm_id+'_' + str(i), 'X':random.randint(0,100),'Y':random.randint(0,100), 'Trust':round(random.uniform(0.5,1.0),3), 'RAM':random.uniform(0.1, 4.0)}, ignore_index = True)

    ran_comp_res = Community.generate_comp_resource(self)

    for i in ran_comp_res:
      val_class = ran_comp_res[i]
      for j in val_class:
        df.at[j, 'Comp_Res'] = i

    return df

  # fungsi untuk generate computation resource
  def generate_comp_resource(self):
    val = self.number_of_node
    if (val > 3):
      ls_rand_gen = [0.1, 0.15, 0.35, 0.3]
      ls_ha = []

      if (val > 4):
        for i in ls_rand_gen:
          ls_ha.append(int(np.ceil(i * val)))

        if (sum(ls_ha) > val):
          ls_ha[2] -= (sum(ls_ha) - val)
        elif (sum(ls_ha) < val):
          ls_ha[2] += (val - sum(ls_ha))

        if (ls_ha[3] > ls_ha[2]):
          ls_ha[2], ls_ha[3] = ls_ha[3], ls_ha[2]
      elif (val == 4):
        ls_ha = [1,1,1,1]

      dict_res = {}
      ls_index = range(val)
      for i,val in enumerate(ls_ha):
        res_random = random.sample(ls_index, val)
        dict_res['Class-'+str(i+1)]  = res_random
        ls_index = list(set(ls_index)-set(res_random))

      return dict_res

    elif (val == 3):
      dict_res = {
          'Class-1':[0],
          'Class-2':[1],
          'Class-3':[2]
      }
 
      return dict_res

    elif (val == 2):
      dict_res = {
          'Class-1':[0],
          'Class-2':[1],
      }

      return dict_res

  # fungsi untuk membuat matrik relasi node
  def generate_matrix_relation(self):
    number_node = self.number_of_node
    prob_connection = self.prob_connection
    mat_relation = np.identity(number_node) 
    all_con_posible = number_node -1

    if (prob_connection == 0):
      new_prob_con = []
      
      for i in range(number_node):
        new_prob_con.append(random.random())
      prob_connection = new_prob_con
    

    for i in range(number_node):
      list_row = list(mat_relation[i,:])

      list_pos_con = []
      list_already_con = []

      for n,val in enumerate(list_row):
        if val == 0:
          list_pos_con.append(n)
        elif val == 1 and n != i:
          list_already_con.append(n)

      number_con = int(np.ceil(prob_connection[i] * all_con_posible))
      random_con = 0

      if (len(list_already_con) < number_con):
        con_left = number_con - (all_con_posible - len(list_pos_con))
        rand_con = random.sample(list_pos_con, con_left)
        
        for j in rand_con:
          mat_relation[j,i] = 1
          mat_relation[i,j] = 1

    self.matrix_relation = mat_relation
    return mat_relation
  
  # menambahkan quality pada matrik connection
  def add_quality_relation(mat_relation):
    mat_copy = mat_relation.copy()
    for i in range(len(mat_relation)):
      for n,val in enumerate(list(mat_relation[i,:])):
        if (n > i and val == 1):
          rand = round(random.uniform(0.5,1.0),3)
          mat_copy[n,i] = rand
          mat_copy[i,n] = rand    
    return mat_copy 

  # fungsi untuk generate tabel yang berisi daftar koneksi tiap node
  def generate_table_relation(self):
    number_node = self.number_of_node
    mat_relation = Community.generate_matrix_relation(self)

    mat_quality_relation = Community.add_quality_relation(mat_relation)

    df = pd.DataFrame(columns=['node_1','node_2','quality','datetime'])

    for i in range(number_node):
      list_node_rel = list(mat_quality_relation[i,:])
      for n,val in enumerate(list_node_rel):
        if (n != i and val != 0):
          df = df.append({'node_1':'A_'+ str(i), 'node_2':'A_' + str(n), 'quality':val,'datetime':dt.datetime.now()}, ignore_index = True)

    return df

  #fungsi untuk kalkulasi
  def calculate_table(self, alpha=0.3, beta=0.3, gamma=0.4):
    start_time = time.time()
    df_calculate = self.table_of_relation.groupby(['node_1']).sum()
    df_calculate['relation'] = list(self.table_of_relation.groupby(['node_1']).size())
    df_calculate['C_score'] = df_calculate['quality']/df_calculate['relation']
    
    df_node = self.table_of_node.copy()
    df_node['comp_rate'] = df_node['RAM'].apply(Community.parse_ram)
    df_node['resource_rate'] = df_node['Comp_Res'].apply(Community.parse_comp_res)

    df_calculate['comp_rate'] = df_node['comp_rate'].tolist()
    df_calculate['resource_rate'] = df_node['resource_rate'].tolist()
    df_calculate['T_score'] = self.table_of_node['Trust'].tolist()
    df_calculate['I_score'] = (df_calculate['comp_rate']+df_calculate['resource_rate'])/2
    df_calculate['S_score'] = (alpha*df_calculate['C_score'])+(beta*df_calculate['I_score'])+(gamma*df_calculate['T_score'])

    result = df_calculate.sort_values(by='S_score', ascending=False)
    
    print("Proses perhitungan: %s second " % ((time.time() - start_time)))

    return result

  def parse_ram(ram):
    pwr = 0
    if ram < 1 :
        pwr = 0.3
    if ram >= 1 :
        pwr = 0.6
    if ram >=4 :
        pwr = 1
    return pwr

  def parse_comp_res(class_type):
    val = 0
    if (class_type == 'Class-1'):
      val = 0.8
    elif (class_type == 'Class-2'):
      val = 0.6
    elif (class_type == 'Class-3'):
      val = 0.4
    elif (class_type == 'Class-4'):
      val = 0.2
    
    return val