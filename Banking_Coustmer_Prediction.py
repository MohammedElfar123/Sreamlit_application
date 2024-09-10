import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import pickle

# Load model

Coustomer_Churn_Pre = 'Coustomer_Churn_Prediction.sav'
model = pickle.load(open(Coustomer_Churn_Pre, 'rb'))


# # Load data
data= pd.read_csv("Bank Customer Churn Prediction.csv")


# Define page options
options = st.sidebar.selectbox("Choose Page", ["Prediction", "EDA", "View Dataset"])

# Prediction Page
if options == "Prediction":
    st.title("Banking Customer Churn Prediction")
    st.markdown("This app predicts whether a customer will churn or not based on various customer attributes.")
    
    # Input fields 
    credit_score = st.number_input("**Inter Credit Score:**  Min-Max: (383-850) ", min_value= 383, max_value= 850, value= 398)
    country = st.number_input("**Inter Country:**  (0 for France, 1 for Spain, 2 for Germany)", min_value= 0, max_value= 2, value= 2)
    
    gender = st.number_input("**Inter Gender:**   (0 for Male, 1 for Female) ", min_value= 0, max_value= 1, value= 1)
    age = st.number_input("**Inter Age:**  Min-Max: (18-92)", min_value= 18, max_value= 100, value= 18)
    
    tenure = st.number_input("**Inter Tenure:**  Min-Max: (0, 10) " , min_value=0, max_value= 10, value= 4)
    balance = st.number_input("**Inter Balance:**  Min-Max: (0-250898.09) ", min_value =0.0, max_value= 250898.09 , value= 10000.00, step= 1.00 )
    
    products_number = st.number_input("**Inter Number of Products:**  Min-Max: (1-4) ", min_value=1, max_value=4, value=1)
    credit_card = st.number_input("*Has Credit Card :** (0 for No, 1 for Yes)", min_value= 0, max_value= 1, value= 0)

    active_member = st.number_input("**Is Active Member:** (0 for No, 1 for Yes) ", min_value=0, max_value=1, value=1)
    estimated_salary = st.number_input("**Estimated Salary:**  Min-Max: (11.58-199992.48) ", min_value= 11.58, max_value= 199992.48, value= 10000.00 ,step = 1.00, format= "%.2f")
    
    # Create input DataFrame
    input_data = pd.DataFrame([[credit_score, country, gender, age, tenure, balance, products_number, credit_card, active_member, estimated_salary]],
                              columns=['credit_score', 'country', 'gender', 'age', 'tenure', 'balance', 'products_number', 'credit_card', 'active_member', 'estimated_salary'])

      
# Display the prediction button and result
if st.button('Predict'):
    # Make the prediction when the button is clicked
    prediction = model.predict(input_data)  
    
    # Check the prediction value
    if prediction[0] == 1: 
        st.markdown(f"**The model prediction is:** Churn")
        st.markdown("**Reason:** This customer is likely to churn based on their characteristics.")
    else:
        st.markdown(f"**The model prediction is:** No Churn")
        st.markdown("**Reason:** This customer is likely to stay with the company.")
        
#-------------------------------------------------------------------------------------------------------------------------------------------------------------
    
#   (EDA Page) #

elif options == "EDA":
    st.title("Exploratory Data Analysis")
    
    # Basic Stats
    st.markdown("### Dataset Summary")
    st.write(data.describe().round(2))  

    # Churn Distribution Pie Chartig
    st.markdown("### Churned vs Not Churned")
    sizes = [data.churn[data['churn'] == 1].count(), data.churn[data['churn'] == 0].count()]
    labels = ['Churned', 'Not Churned']
    
    fig_pie = go.Figure(data=[go.Pie(labels=labels, values=sizes, hole=0.3)])  
    fig_pie.update_layout(title_text="Churned VS Not Churned", title_font_size=20)
    fig_pie.update_traces(textposition='inside', textinfo='percent+label')
    st.plotly_chart(fig_pie)

    # Loop to plot distributions and outliers for numeric columns
    st.markdown("### Distribution and Outliers for Numeric Columns")
    
    numeric_columns = ['credit_score', 'age', 'tenure', 'balance', 'products_number', 
                       'credit_card', 'active_member', 'estimated_salary', 'churn']
    
    for col in numeric_columns:
        st.markdown(f"#### Distribution of {col}")
        fig_hist = px.histogram(data, x=col, text_auto=True, width=1400, title=f"Distribution of {col}")
        st.plotly_chart(fig_hist)

        st.markdown(f"#### Box Plot of {col}")
        fig_box = px.box(data, x=col, points="all", width=1400, title=f"Box Plot of {col}")
        st.plotly_chart(fig_box)

    # Categorical columns distribution
    st.markdown("### Distribution for Categorical Columns")
    categorical_columns = ["country", "gender"]
    
    for col in categorical_columns:
        st.markdown(f"#### Distribution of {col}")
        fig_cat = px.histogram(data, x=col, title=f"Distribution of {col}", template="simple_white", text_auto=True)
        st.plotly_chart(fig_cat)
        
    st.markdown("### Correlation Heatmap")
    corr_matrix = data.select_dtypes("number").corr()
    fig_corr = px.imshow(corr_matrix, 
                         text_auto=True, 
                         aspect="auto", 
                         color_continuous_scale='Viridis', 
                         title="Correlation Heatmap")
    st.plotly_chart(fig_corr)

#-------------------------------------------------------------------------------------------------------------------------------------------------------------

#   Dataset Page #
elif options == "View Dataset":
    st.title("View Dataset")
    st.dataframe(data)  
    

    st.markdown("### Dataset Information")
    
    # Capture the output of data.info()
    import io 
    buffer = io.StringIO()
    data.info(buf=buffer)
    info_output = buffer.getvalue()  # Get the string content from the buffer

    # Display the captured info in Streamlit
    st.text(info_output)

