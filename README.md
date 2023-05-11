# GridSearch

:construction: Projeto em construção :construction:

Dependencies
<h4 align="center"> 
  sklearn
  keras
  pickle
  pandas
  numpy
  itertools
<\h4>

Para executar o projeto, você deve:
<h4 align="center"> 
  Alterar os paths no script de manage_results, para que os resultados sejam
  escritos na pasta "results"
<\h4>
A execução base é feita da seguinte forma:

Defina um dicionário python com as informações: path do arquivo csv(tabela com os dados);
                                                lista de string das variáveis do seu modelo (como escritos no csv);
                                                string que identifica qual a variável target do seu modelo;
                                                string que serve como identificador do teste nos resultados do seu modelo.
  
input_data = {'path': '/home/trama/Documents/Python Scripts/grid_search/datasets/dataset_exemplo',
            'variables': [<var1>,<var2>,<var3>,...,<var_n>],
            'target': <taget_variable>,
            'label' : <some_label_to_identify>
           }

Em seguida, import o arquivo "grid_search.py" para o seu ambiente ( Jupyter notebook, por exemplo ), 
e execute o comando: grid_search.init_process(input_data,<sep>)
Onde <sep> define o separador do seu arquivo csv, como exemplo: ',' ou ';'.
