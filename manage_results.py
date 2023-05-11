import pickle


#To save result
def save_best_result(dict_and_score_result,label,idx):
        
    arq_results = open('/home/trama/Documents/Python Scripts/grid_search/results/results_'+label+'_'+f'{idx}.bin','wb')

    pickle.dump(dict_and_score_result,arq_results)

    arq_results.close()

    return False


#To see result
def open_result(arq_path): #set file path

    arq = open(arq_path,'rb')

    print('\n')    
    while True:
        try:
            result = pickle.load(arq)
            print(f'config: {result[0]}\nWith score: {result[1]}',end='\n\n')

        except EOFError:
            arq.close()
            break
