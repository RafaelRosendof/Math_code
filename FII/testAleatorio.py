import fundamentus as fu 
import pandas as pd 
import numpy as np 
import yfinance as yf

def test_fundamentos():
    df = fu.get_detalhes_papel("BBSE3")
    
    print(f"Detalhando a função get_detalhes_papel() 1 \n\n {type(df)}")
    
    print(f"Detalhamento das colunas de df {df.head()}\n\n")
    
    df2 = fu.get_detalhes_raw()
    print(f"Detalhando a função get_detalhes_raw() 2 \n\n {type(df)}")
    
    print(f"Detalhamento das colunas de df {df.head()}\n\n")
    
    print("\n\n\n-------------------------------------------------------------\n\n\n")
    
    df3 = fu.get_papel("BPML11")
    print(f"Detalhando a função get_papel() 3 \n\n {type(df)}")
    
    print(f"Detalhamento das colunas de df \n\n {df.head()}\n\n")
    print("\n\n\n-------------------------------------------------------------\n\n\n")
    df4 = fu.get_resultado()
    print(f"Detalhando a função get_resultado 4 \n\n {type(df4)}")
    
    print(f"Detalhamento das colunas do dataframe df \n\n {df4}\n\n")
    print("\n\n\n-------------------------------------------------------------\n\n\n")
    df5 = fu.list_papel_all()
    print(f"Detalhando a função get_detalhes_papel_all() 5 \n\n {type(df5)}")
    
    print(f"Detalhamento de todos os papéis  {df5[:5]}\n\n")
    
    return df,df2,df3,df4,df5 

def test_scrap_b3():
    pass

def test_data_yahoo():
    pass 

def main():
    
    df,df2,df3,df4,df5 = test_fundamentos()
    
    print("\n\n Feito o teste da biblioteca de fundamentos \n\n")
    
if __name__ == "__main__":
    main()