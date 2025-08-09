# 🌱 Aplicativo para Cálculo do Balanço Hídrico Climatológico (BHC)

Este aplicativo, desenvolvido em **Python** com **Streamlit**, calcula o **Balanço Hídrico Climatológico** usando dados médios mensais de temperatura e precipitação do **WorldClim** e/ou dados inseridos manualmente.

O cálculo é feito pelo método de **Thornthwaite & Mather (1955)**, com ajuste do fotoperíodo e CAD (Capacidade de Água Disponível do solo).  
Ele gera tabela completa com todos os componentes do balanço (ARM, ALT, ETR, DEF, EXC), gráficos e exportação para Excel.

---

## 👨‍🏫 Autores

- **Claudio Ricardo da Silva** – Universidade Federal de Uberlândia (UFU)  
- **Nildo da Silva Dias** – Universidade Federal Rural do Semi-Árido (UFERSA)

---

## 🚀 Como executar localmente

1. Clone o repositório:
   ```bash
   git clone https://github.com/rickpira/bhc.git
   cd bhc
   ```

2. Crie um ambiente virtual (opcional, mas recomendado):
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # Linux/Mac
   .venv\Scripts\activate     # Windows
   ```

3. Instale as dependências:
   ```bash
   pip install -r requirements.txt
   ```

4. Rode o aplicativo:
   ```bash
   streamlit run bhc4.py
   ```

---

## 🌐 Executar no Streamlit Cloud

O app também está disponível online (sem precisar instalar nada localmente):

🔗 **[Abrir no Streamlit Cloud](https://balancohidrico.streamlit.app/)**

---

## 📦 Estrutura dos arquivos

- `bhc4.py` → Código principal do aplicativo
- `requirements.txt` → Dependências para rodar no Streamlit Cloud
- `.python-version` → Define a versão Python para compatibilidade
- `readme.md` → Este arquivo de documentação

---

## 📊 Funcionalidades

- Download automático dos dados climáticos médios do **WorldClim** (10 arc-min)
- Cálculo do **ETo** pelo método de **Thornthwaite**
- Ajuste do cálculo pelo **fotoperíodo mensal**
- Inserção manual de dados (opcional)
- Cálculo automático de:
  - ARM – Armazenamento de água no solo
  - ALT – Alteração no ARM
  - ETR – Evapotranspiração real
  - DEF – Déficit hídrico
  - EXC – Excedente hídrico
- Geração de **gráficos interativos**
- Exportação do resultado em **Excel (.xlsx)**

---

## 📜 Licença

Este projeto está licenciado sob a licença **MIT** – sinta-se à vontade para usar e modificar, mantendo os créditos originais.
