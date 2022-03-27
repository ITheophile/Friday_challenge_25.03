import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

import plotly.express as px


# Load data
df = pd.read_csv("https://raw.githubusercontent.com/chriswmann/datasets/master/500_Person_Gender_Height_Weight_Index.csv")
df['Index'] = df['Index'].map({0:'Extremely Weak', 1:'Weak', 2:'Normal', 3:'Overweight', 4:'Obesity', 5:'Extreme Obesity'})



# nb_weight_men = df[df['Gender'] == 'Male']['Index'].value_counts()
# nb_weight_women = df[df['Gender'] == 'Female']['Index'].value_counts()

# nb_weight_men = pd.DataFrame({'BMI': nb_weight_men.index, 'Counts': nb_weight_men.values})
# nb_weight_women = pd.DataFrame({'BMI': nb_weight_women.index, 'Counts': nb_weight_women.values})

# App Title

st.title('Welcome to Friday Challenge!')
selected_task = st.sidebar.selectbox('TASK', ['Rotation Matrix', 'Data Overview', 'Weight vs Height', 'Histograms', 'Euclidian Distance', 'PCA'])







if selected_task == 'Rotation Matrix':
    # Rotate vector

    def rotate_vector(vector, angle):
        
        # conversion from degree to radians
        angle_radians = (angle * np.pi) / 180

        # from list to ndarray
        vector = np.array(vector)

        # matrix for the rotation
        rotation_matrix = np.array([ [np.cos(angle_radians), - np.sin(angle_radians)],
                                    [np.sin(angle_radians),   np.cos(angle_radians)] ])

        # rotated vector
        transf_vector = rotation_matrix.dot(vector)
    
        x_origin, y_origin = vector[0], vector[1]
        x_transf, y_transf = transf_vector[0], transf_vector[1]
        
        fig, ax = plt.subplots(figsize = (8, 8))
        ax.arrow(0, 0, x_transf, y_transf, width = 0.3,head_width = 0.8, facecolor = 'r', edgecolor = 'r', label = 'transformed_vector')
        ax.arrow(0, 0, x_origin, y_origin, width = 0.3, head_width = 0.8, facecolor = 'g', edgecolor = 'g', label ='original_vector')
        

        limit = np.abs(vector).max()
        ax.set_xlim(-2*limit, 2*limit)
        ax.set_ylim(-2*limit, 2*limit)
        ax.set_xlabel('x axis')
        ax.set_ylabel('y axis')
        ax.set_title(f'Vector rotation of {angle}°')
        ax.legend()
        
        return fig


    st.subheader('Demo of Vector Rotation')
    x = st.sidebar.number_input('Enter x coordinate: ', 10)
    y = st.sidebar.number_input('Enter y coordinate', 0)

    angle = st.sidebar.slider("Enter an angle: ", 0, 360, step=10)

    
    st.write(rotate_vector([x, y], angle))





elif selected_task == 'Data Overview':

    show_data = st.sidebar.checkbox('Show data')
    

    st.subheader('Body Mass Index data')

    if show_data:
        st.write(df)


    # nb_weight_men = df[df['Gender'] == 'Male']['Index'].value_counts()
    # nb_weight_women = df[df['Gender'] == 'Female']['Index'].value_counts()

    # nb_weight_men = pd.DataFrame({'BMI': nb_weight_men.index, 'Counts': nb_weight_men.values})
    # nb_weight_women = pd.DataFrame({'BMI': nb_weight_women.index, 'Counts': nb_weight_women.values})

    # fig1 = px.bar(
    #     nb_weight_men, x = "BMI", y = "Counts",
    #     template = 'seaborn',
    #     title = 'Men' )

    # st.plotly_chart(fig1) 

    # fig2 = px.bar(
    #     nb_weight_women, x = "BMI", y = "Counts",
    #     template = 'seaborn',
    #     title = 'Women' )

    # st.plotly_chart(fig2) 

    



elif selected_task == 'Weight vs Height':
    nb_weight_men = df[df['Gender'] == 'Male']
    nb_weight_women = df[df['Gender'] == 'Female']

    fig = px.scatter(data_frame=df, x = 'Weight', y = 'Height', color= 'Index',
    template = 'seaborn',
        title = 'All people' )


    st.plotly_chart(fig)


    fig1 = px.scatter(data_frame=nb_weight_men, x = 'Weight', y = 'Height', color= 'Index',
    template = 'seaborn',
        title = 'Men' )


    st.plotly_chart(fig1)


    fig2 = px.scatter(data_frame=nb_weight_women, x = 'Weight', y = 'Height', color= 'Index',
    template = 'seaborn',
        title = 'Women' )


    st.plotly_chart(fig2)






elif selected_task == 'Histograms':
    nb_weight_men = df[df['Gender'] == 'Male']
    nb_weight_women = df[df['Gender'] == 'Female']
    
    feature  = st.sidebar.radio('Feature', ['Height', 'Weight'])

    if feature == 'Height':
        fig, ax = plt.subplots()
        ax.hist(df['Height'], bins= 50)
        ax.set_title('All')
        ax.set_xlabel('Height')
        
        st.plotly_chart(fig)



        fig1, ax = plt.subplots()
        ax.hist(nb_weight_men['Height'], bins= 50)
        ax.set_title('Men')
        ax.set_xlabel('Height')
        
        st.plotly_chart(fig1)


        fig2, ax = plt.subplots()
        ax.hist(nb_weight_women['Height'], bins= 50)
        ax.set_title('Women')
        ax.set_xlabel('Height')
        
        st.plotly_chart(fig2)



    elif feature == 'Weight':
        fig, ax = plt.subplots()
        ax.hist(df['Weight'], bins= 50)
        ax.set_title('All')
        ax.set_xlabel('Weight')
        
        st.plotly_chart(fig)



        fig1, ax = plt.subplots()
        ax.hist(nb_weight_men['Weight'], bins= 50)
        ax.set_title('Men')
        ax.set_xlabel('Weight')
        
        st.plotly_chart(fig1)


        fig2, ax = plt.subplots()
        ax.hist(nb_weight_women['Weight'], bins= 50)
        ax.set_title('Women')
        ax.set_xlabel('Weight')
        
        st.plotly_chart(fig2)



elif selected_task == 'Euclidian Distance':
    fig, ax = plt.subplots()
    index_mean= df.groupby(['Index']).mean()
    ax.scatter(x=index_mean['Weight'], y=index_mean['Height'])



    # px.scatter()
    
    st.pyplot(fig)