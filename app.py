import dash
from dash import dcc
import dash_bootstrap_components as dbc
from dash import html
from dash.exceptions import PreventUpdate
from dash.dependencies import Input, Output, State
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
from dash_bootstrap_templates import load_figure_template
import copy
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.cross_decomposition import PLSRegression, PLSSVD

import dash_mantine_components as dmc
from dash_iconify import DashIconify

import io
import zipfile

# load_figure_template(["QuartZ"])
#load_figure_template(["Sketchy"])
load_figure_template(["United"])
load_figure_template(["Journal"])
load_figure_template(["superhero"])



import base64
from io import BytesIO
from plotly.subplots import make_subplots

import json

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

###################################################################################################################################################################################
# IMPORTANDO OS DADOS:

# B=pd.read_excel('DadosGraficosbonitos.xlsx',header = 5)


# tags = ['Vazão mássica de H2o (kg/h)','Vazão mássica de EtOH(kg/h)','Temperatura do combustor (K)']
# dados = B[tags]
# dados_filter = copy.deepcopy(dados)
# dados_filter = dados_filter[dados_filter[tags[-1]]>825]

# ENT = 5


#---
# excel_icon = html.I(className='far fa-file-excel', style=dict(display='inline-block'))


# ----------------- Encoder <---> Decoder ---------------------------#

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        dtypes = (np.datetime64, np.complexfloating)
        if isinstance(obj, dtypes):
            return str(obj)
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            if any([np.issubdtype(obj.dtype, i) for i in dtypes]):
                return obj.astype(str).tolist()
            return obj.tolist()
        return super(NpEncoder, self).default(obj)
def NpDecoding(json_s, classes):
    keys_i = list(json_s.keys())
    for i in range(len(json_s.keys())):
        class_i = classes[i]
        if class_i == "<class 'numpy.int64'>":
            json_s[keys_i[i]] = np.int64(json_s[keys_i[i]])
        elif class_i == "<class 'numpy.ndarray'>":
            json_s[keys_i[i]] = np.array(json_s[keys_i[i]])
    return json_s
#--------------------------------------------------------------------#

excel_icon = html.I(className='fas fa-skull', style=dict(display='inline-block'))
btn_text=html.Div('Baixar parâmetros', style=dict(paddingRight='0.5vw',display='inline-block'))
btn1_content = html.Span([btn_text,excel_icon])
#---

def get_model(ENT):
    OR = str(ENT)
    if OR == '2':
        NC = 6
        trans = lambda x1,x2: (1,x1,x2,x1*x1,x2*x2,x1*x2)
    if OR == '3':
        NC = 10
        trans = lambda x1,x2: (1,x1,x2, x1*x1, x2*x2, x1*x2, x1*x1*x1, x2*x2*x2, x1*x1*x2, x2*x2*x1)
    if OR == '4':
        NC = 15
        trans = lambda x1,x2: (1,x1,x2, x1*x1, x2*x2, x1*x2, x1*x1*x1, x2*x2*x2, x1*x1*x2, x2*x2*x1,x1*x1*x1*x1, x2*x2*x2*x2, x1*x1*x1*x2, x2*x2*x2*x1, x1*x1*x2*x2)
    if OR == '5':
        NC = 22
        trans = lambda x1,x2: (1,x1,x2, x1*x1, x2*x2, x1*x2, x1*x1*x1, x2*x2*x2, x1*x1*x2, x2*x2*x1,x1*x1*x1*x1, x2*x2*x2*x2, x1*x1*x1*x2, x2*x2*x2*x1, x1*x1*x2*x2,
              x1*x1*x1*x1*x1, x1*x1*x1*x1*x2,x1*x1*x1*x2*x2,x1*x2*x2*x2*x2,x1*x1*x2*x2*x2,x2*x2*x2*x2*x2,x1*x2*x2*x2*x2)

    return trans, NC

# trans,NC = get_model(ENT)

def get_best_NC(dados,trans,NC):

    C=np.zeros((dados.shape[0],NC))

    for i in range (dados.shape[0]):
        x1=dados.iloc[i,0]
        x2=dados.iloc[i,1]
        C[i,:] = trans(x1,x2)

    # Calculate MSE using cross-validation, adding one component at a time
    RMSE = np.zeros((NC-1,1))
    for i in np.arange(1, NC-1):
        pls2 = PLSRegression(n_components=i,scale=False)
        pls2.fit(C, dados.iloc[:,2])
        Y_pred = pls2.predict(C)
        RMSE[i-1] = mean_squared_error(dados.iloc[:,2],Y_pred,squared=False)
        
    return RMSE,C

# RMSE,C = get_best_NC(dados,trans,NC)



def get_model_PLS(RMSE,dados,C):

    NC2 = np.argmin(RMSE)
    pls2 = PLSRegression(n_components=NC2,scale=True)
    pls2.fit(C, dados.iloc[:,2])
    Y_pred = pls2.predict(C)
    
    return Y_pred,pls2

# Y_pred, pls2 = get_model_PLS(RMSE,dados,C)

def get_surface(dados,trans, pls2, pls3,LIM):

    A = copy.deepcopy(dados)

    
    
    
    for c in range (1,A.shape[0]):
        if A.iloc[c,0] != A.iloc[c-1,0]:
            dif1=abs(A.iloc[c,0]-A.iloc[c-1,0])
            dif1=round(dif1,4)
            break
        
    for c in range (1,A.shape[0]):
        if A.iloc[c,1] != A.iloc[c-1,1]:
            dif2=abs(A.iloc[c,1]-A.iloc[c-1,1])
            dif2=round(dif2,4)
            break
        
#     N1=(A.iloc[:,0].max()-A.iloc[:,0].min())/(dif1)
#     N2=(A.iloc[:,1].max()-A.iloc[:,1].min())/(dif2)
    N1 = 200
    N2 = 200

    var1=np.linspace(A.iloc[:,0].min(),A.iloc[:,0].max(),round(N1))
    var2=np.linspace(A.iloc[:,1].min(),A.iloc[:,1].max(),round(N2))

    Fhat=np.zeros((round(N1),round(N2)))
    Ccalc=np.zeros((np.shape(Fhat)[1],np.shape(Fhat)[0]))
    Ccalc_r=np.zeros((np.shape(Fhat)[1],np.shape(Fhat)[0]))
    for i in range (np.shape(Fhat)[1]):
        for j in range (np.shape(Fhat)[0]):
            Pdavez=var2[i]
            Tdavez=var1[j] 
            predict_ = pls2.predict(np.array(trans(Tdavez,Pdavez)).reshape(1,-1))
            predict_2 = pls3.predict(np.array(trans(Tdavez,Pdavez)).reshape(1,-1))
            Ccalc[i,j] = predict_
            Ccalc_r[i,j] = predict_
            
            
            if LIM != None and predict_2<LIM:
                Ccalc_r[i,j] = np.nan
            
    return Ccalc,Ccalc_r, var1, var2

# Ccalc, var1, var2 = get_surface(dados,trans,pls2)


def blank_fig():
    fig = go.Figure(go.Scatter(x=[], y = []))
    fig.update_layout(template = "superhero")
    fig.update_xaxes(showgrid = False, showticklabels = False, zeroline=False)
    fig.update_yaxes(showgrid = False, showticklabels = False, zeroline=False)
    fig.update_layout(
                    margin=dict(l=20, r=20, b=35, t=25)) 
    
    return fig


def figure1(RMSE):
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x = np.linspace(1,RMSE.shape[0],RMSE.shape[0]-1),
        y = RMSE.reshape((-1,)),
        name = "Número de componentes",
        mode = 'markers',
        marker_color = 'rgba(255,182,193,.9)'
    ))

    fig.update_traces(mode = 'markers',marker_line_width = 2, marker_size = 10)

    fig.update_layout(xaxis_title = "Número de componentes",
                 yaxis_title = "RMSE") 
    
    return fig






def figure2(dados,Y_pred):
    fig = go.Figure()


    fig.add_trace(go.Scatter(
        y = dados.iloc[:,2].values.reshape((-1,)),
        name = "Real",
        mode = 'markers',
        marker_color = '#6951e2',
        marker_size = 12
    ))

    fig.add_trace(go.Scatter(
        y = Y_pred.reshape((-1,)),
        name = "Predito",
        mode = 'markers',
        marker_symbol="x",
        marker_color = '#eb6ecc',
        marker_size = 8
    ))


    fig.update_traces(mode = 'markers',marker_line_width = 1)

    fig.update_layout(xaxis_title = "Amostra",
                yaxis_title = "Variável") 
    return fig


def figure3(Ccalc,var1,var2,Ccalc_r,dados):

    

    cmap = plt.get_cmap("tab10")
    colors_saddle = np.zeros(shape=Ccalc_r.shape)  


    fig = make_subplots(rows=1,cols=1)
    # fig = go.Figure()


    fig.add_trace(go.Surface(z=Ccalc, x=var1, y=var2, surfacecolor = colors_saddle, opacity = 0.5, showscale = False))


    fig.add_trace(go.Surface(z=Ccalc_r, x=var1, y=var2, colorbar = {"orientation": "v", "x":0.9, "xanchor":"right"}))
    # fig.add_trace(go.Surface(z=Ccalc_r, x=var1, y=var2, colorscale = 'Jet', colorbar = {"orientation": "v", "x":0.9, "xanchor":"right"}))

    fig.update_traces(contours_z=dict(show=True, usecolormap=True,
                                    highlightcolor="limegreen", project_z=True))
    fig.update_layout(autosize=True, scene = dict(
                        xaxis = dict(title=dados.columns.to_list()[0],
                            backgroundcolor="rgb(200, 200, 230)",
                            gridcolor="white",
                            showbackground=True,
                            zerolinecolor="white",),
                        yaxis = dict( title=dados.columns.to_list()[1],
                            backgroundcolor="rgb(230, 200,230)",
                            gridcolor="white",
                            showbackground=True,
                            zerolinecolor="white"),
                        zaxis = dict(title=dados.columns.to_list()[2],
                            backgroundcolor="rgb(230, 230,200)",
                            gridcolor="white",
                            showbackground=True,
                            zerolinecolor="white",),),
                            height=750,
                    margin=dict(l=5, r=5, b=5, t=5)) 



    return fig
####################################################################################################################################################################################

def cabecalho(app):
    title = html.Div( id = 'oi-id',
        style = {"textAlign":"center"},
        children = [
            html.H1(
        'Módulo de visualização de superfícies',
        style = {"marginTop":20,"marginLeft":"10px"}
    )
        ]
    )

    info_about_app = html.Div(
        style = {"textAlign":"center"},
        children = [
            html.H3(
        'Gere superfícies bonitas interpolando via PLS',
        style = {"marginLeft":"10px"}
    )
        ]
    )

    logo_image = html.Img(
        src = app.get_asset_url("Logo_ISI.png"), style = {"float":"right","height":80,"marginTop":20}
    )

    link = html.A(logo_image,href="https://senaicetiqt.com/inovacao/")

    return dbc.Row([
        dbc.Col(width = 3),
        dbc.Col([dbc.Row([title]), dbc.Row([info_about_app])],width = 6),
        dbc.Col(link, width = 3)
    ])

info_box = dbc.Card(className='card secundary mb-3',
                    children =[
                        dbc.CardHeader(
                            "SetUp",
                            style = {
                                "text-align": "center",
                                "color": "white",
                                "border-radius":"1px",
                                "border-width":"5px",
                                "border-top":"1 px solid rgb(216, 216, 216)"
                            }
                        ),
                        dbc.CardBody(
                            [
                                html.Div([
                                    # dbc.Button('Carregar',id = 'open'),
                                    dmc.Button("Carregar dados",id='open',
                                                leftIcon=DashIconify(icon='fa-solid:file-upload'), size="md",
                                                color='#23a847',variant="gradient", gradient={"from": "teal", "to": "lime", "deg": 105}),
                                    dbc.Modal([
                                        dbc.ModalHeader("Carregar Arquivo Excel"),
                                        dbc.ModalBody([
                                            dcc.Upload(
                                                id="upload-data",
                                                children=html.Div([
                                                    "Arraste e solte ou ",
                                                    html.A("selecione um arquivo Excel")
                                                ]),
                                                multiple=False,
                                            ),
                                            html.Div(id="output-data-upload"),
                                        ]),
                                        dbc.ModalFooter([
                                            dbc.Button("Fechar", id="close", className="btn btn-primary")
                                        ]),
                                    ],
                                    id="modal",
                                    size="lg"
                                    ),
                                    # dcc.Store stores the intermediate value
                                    dcc.Store(id='intermediate-value')]
                                )
                            ]
                        )
                    ])


get_variable = dbc.Card(className='card secundary mb-3',
                    children =[
                        dbc.CardHeader(
                            "Selecione as variáveis do modelo",
                            style = {
                                "text-align": "center",
                                "color": "white",
                                "border-radius":"1px",
                                "border-width":"5px",
                                "border-top":"1 px solid rgb(216, 216, 216)"
                            }
                        ),
                        dbc.CardBody(
                            [
                                html.Div(
                                    children = [
                                        html.Div(
                                            id = 'div-x1-selector',
                                            children = [
                                                html.Label("Selecione o eixo x1"),
                                                           dcc.Dropdown(
                                                               id = 'x1-selector',
                                                               options = [{'label': i, "value":i} for i in range(5)],
                                                               multi = False
                                                           )
                                            ]
                                        ),

                                        html.Div(
                                            id = 'div-x2-selector',
                                            children = [
                                                html.Label("Selecione o eixo x2"),
                                                           dcc.Dropdown(
                                                               id = 'x2-selector',
                                                               options = [{'label': i, "value":i} for i in range(5)],
                                                               multi = False
                                                           )
                                            ]
                                        ),

                                        html.Div(
                                            id = 'div-x3-selector',
                                            children = [
                                                html.Label("Selecione o eixo x3"),
                                                           dcc.Dropdown(
                                                               id = 'x3-selector',
                                                               options = [{'label': i, "value":i} for i in range(5)],
                                                               multi = False
                                                           )
                                            ]
                                        )
                                    ]
                                )
                            ]
                        )
                    ])
    

get_order = dbc.Card(className='card secundary mb-3',
                    children =[
                        dbc.CardHeader(
                            "Selecione a ordem do modelo",
                            style = {
                                "text-align": "center",
                                "color": "white",
                                "border-radius":"1px",
                                "border-width":"5px",
                                "border-top":"1 px solid rgb(216, 216, 216)"
                            }
                        ),
                        dbc.CardBody(
                            [
                                html.Div(
                                    children = [
                                        html.Div(
                                            id = 'Div-order-selector',
                                            children = [
                                                html.Label("Selecione a ordem do interpolador"),
                                                           dcc.Dropdown(
                                                               id = 'order-selector',
                                                               options = [{'label': i, "value":i} for i in range(2,6)],
                                                               optionHeight = 20,
                                                               multi = False
                                                           )
                                            ]
                                        )
                                    ]
                                )
                            ]
                        )
                    ])


get_input = dbc.Card(className='card secundary mb-3',
                    children =[
                        dbc.CardHeader(
                            "Região de viabilidade dos dados",
                            style = {
                                "text-align": "center",
                                "color": "white",
                                "border-radius":"1px",
                                "border-width":"5px",
                                "border-top":"1 px solid rgb(216, 216, 216)"
                            }
                        ),
                        dbc.CardBody(
                            [
                                dbc.Row(
                                    dbc.Col(
                                        html.Div(
                                    children = [
                                        html.Div(
                                            id = 'Div-variable-rest',
                                            children = [
                                                html.Label("Selecione a variável com restrição"),
                                                           dcc.Dropdown(
                                                               id = 'var-rest',
                                                               options = [{'label': i, "value":i} for i in range(2,6)],
                                                               optionHeight = 20,
                                                               multi = False
                                                           )
                                            ]
                                        )
                                    ]
                                )
                                    )
                                ),

                                dbc.Row(
                                    dbc.Col(
                                        html.Div(
                                    children = [
                                        html.Div(
                                            id = 'Div-variable-value',
                                            children = [
                                                    html.Label("Ingresse o valor da restrição"),
                                                    dcc.Input(
                                                               id = 'var-value',
                                                               type = "number",
                                                               placeholder="Entre com o valor mímino"
                                                           ),
                                                        # dbc.Col(
                                                        #     html.Button( 'Calcular',
                                                        #         id = 'botao-carregar',
                                                        #         n_clicks = 0
                                                        #         )
                                                        # )
                                                           
                                            ]
                                        )
                                    ]
                                )
                                    ), style={'marginTop':20}
                                ),

                                dbc.Row(dbc.Col(
                                    html.Div(children=[
                                        html.Div(id='compute-surface-div', 
                                                children=[dmc.Button("Calcular superficie",id='botao-model-figs',n_clicks=0,
                                                leftIcon=DashIconify(icon='cib:mathworks'), size="md",
                                                color='#23a847',variant="gradient", gradient={"from": "teal", "to": "lime", "deg": 105})]
                                                ),
                                        dcc.Store(id='intermediate-model-pls2'),
                                        dcc.Store(id='intermediate-model-pls3'),
                                        dcc.Store(id='intermediate-model-rmse'),
                                        dcc.Store(id='intermediate-model-C_json'),
                                        dcc.Store(id='intermediate-model-Y_pred'),
                                        dcc.Store(id='intermediate-model-classes')
                                        ]
                                            )
                                        ),style={'marginTop':10})

                            ]
                        )
                    ])




get_prediction = dbc.Card(className='card secundary mb-3',
                    children =[
                        dbc.CardHeader(
                            "Predição baseada no regressor",
                            style = {
                                "text-align": "center",
                                "color": "white",
                                "border-radius":"1px",
                                "border-width":"5px",
                                "border-top":"1 px solid rgb(216, 216, 216)"
                            }
                        ),
                        dbc.CardBody(
                            [

                                dbc.Row([
                                    dbc.Col(
                                        html.Div(
                                    children = [
                                        html.Div(
                                            id = 'Div-x1-variable-value',
                                            children = [
                                                dbc.Row(
                                                    [
                                                        dbc.Col(
                                                            html.Label("Ingresse o valor da variável x1")
                                                        ),
                                                        dbc.Col(
                                                            
                                                            dcc.Input(
                                                               id = 'var-x1-value',
                                                               type = "number",
                                                               placeholder="Entre com o valor de x1"
                                                           ))

                                                    ]

                                                )


                                                
                                                           
                                            ]
                                        )
                                    ]
                                )
                                    ,width=6),

                                dbc.Col(
                                        html.Div(
                                    children = [
                                        html.Div(
                                            id = 'Div-x2-variable-value',
                                            children = [
                                                dbc.Row(
                                                    [
                                                        dbc.Col(
                                                            html.Label("Ingresse o valor da variável x2")
                                                        ),
                                                        dbc.Col(
                                                            
                                                            dcc.Input(
                                                               id = 'var-x2-value',
                                                               type = "number",
                                                               placeholder="Entre com o valor de x2"
                                                           ))


                                                    ]

                                                )
         
                                                           
                                            ]
                                        )
                                    ]
                                )
                                    ,width=6)
                                    
                                    ], style={'marginTop':20}
                                )
                            ,

                            dbc.Row( [dbc.Col(

                                            dmc.Button("Calcular predição",id='botao-regressor',n_clicks=0,
                                                leftIcon=DashIconify(icon='fluent:predictions-24-filled'), size="md",
                                                color='#23a847',variant="gradient", gradient={"from": "teal", "to": "lime", "deg": 105})

                                            ,width=6),

                                    dbc.Col(
                                            html.Div(
                                            id = 'Div-z-variable-value',
                                            children = [
                                                dbc.Row(
                                                    [
                                                        dbc.Col(
                                                            html.Label("Valor predito de z: ")
                                                        ),
                                                        dbc.Col(
                                                            
                                                            html.Div(
                                                               id = 'var-z-value',
                                                               children=  dcc.Markdown(r'$\hat{z} = f\left ( x_{1},x_{2} \right)$',mathjax=True),
                                                               disable_n_clicks=True,
                                                             style={"font-weight": "bold"}

                                                           ))


                                                    ]

                                                )
         
                                                           
                                            ]
                                        )
                                            ,width=6)], style={'marginTop':20}),

                        dbc.Row( [dbc.Col(
                            html.Div(id='feasible_markdown',children=[
                                dbc.Alert(
                                [html.H6("Previsão de viabilidade da região!", className="alert-heading"),
                                html.Hr(),
                                dcc.Markdown('''
                                    Calcule primeiro a predição para determinar a viabilidade do estado.
                                ''')], color="secondary",style={'marginTop':1,"marginBottom":1}),
                            ]
                        )

                        )], style={'marginTop':20})




                            


                            ]
                        )
                    ])




get_equation = dbc.Card(className='card secundary mb-3',
                    children =[
                        dbc.CardHeader(
                            "Equação obtida na regressão",
                            style = {
                                "text-align": "center",
                                "color": "white",
                                "border-radius":"1px",
                                "border-width":"5px",
                                "border-top":"1 px solid rgb(216, 216, 216)"
                            }
                        ),
                        dbc.CardBody(
                            [

                                dbc.Row([


                                dbc.Col(
                                        html.Div(
                                    children = [
                                        html.Div(
                                            id = 'Div-equation_markdown',
                                            children= dbc.Row([dbc.Col(width=1,align="center"),
                                                                dbc.Col
                                                               (dcc.Markdown(r'$\hat{y} = \left (\frac{x-\mu }{\sigma }  \right )\cdot \tilde{B}\: +\: B_{0}$'
                                                                ,mathjax=True),width=3,align="center"),
                                                               
                                                               
                                                               
                                                            #    dbc.Col(html.Button( 'Baixar parâmetros',
                                                            #             id = 'botao-parametros',
                                                            #             n_clicks = 0
                                                            #             ),width=6)


                                                            # dbc.Col
                                                            #     (dbc.Button(children=btn1_content,id='excel-button',n_clicks=0,
		                                                    #         style=dict(textAlign='center')))

                                                            dbc.Col(html.Div([
                                                                dmc.Button(
                                                                    "Baixar parâmetros",
                                                                    id='parameters-download-btn',
                                                                    n_clicks=0,
                                                                    leftIcon=DashIconify(icon='vscode-icons:file-type-excel'), size="md",
                                                                variant="gradient", gradient={"from": "orange", "to": "red"}),
                                                                dcc.Download(id="download-xls")]),width=3,align="center"),
                                                            
                                                            dbc.Col(width=1,align="center"),

                                                            dbc.Col(html.Div([
                                                                dmc.Button(
                                                                    "Baixar figuras",
                                                                    id='figures-download-btn',
                                                                    n_clicks=0,
                                                                    leftIcon=DashIconify(icon='teenyicons:png-solid'), size="md",
                                                                variant="gradient", gradient={"from": "orange", "to": "red"}),
                                                                dcc.Download(id="download-pngs")]),width=3,align="center"),
                                                            
                                                            dbc.Col(width=1,align="center")



                                                                        
                                                                        ],justify="around",align="center")

                                        )
                                    ]
                                )
                                    ,width=12)
                                    
                                    ], style={'marginTop':20},justify="around"
                                )
                            
                    ])
                    ])








# app = dash.Dash(__name__,external_stylesheets=[dbc.themes.QUARTZ,dbc.themes.BOOTSTRAP,dbc.icons.FONT_AWESOME])
#app = dash.Dash(__name__,external_stylesheets=[dbc.themes.SKETCHY,dbc.themes.BOOTSTRAP,dbc.icons.FONT_AWESOME])
app = dash.Dash(__name__,external_stylesheets=[dbc.themes.BOOTSTRAP,dbc.icons.FONT_AWESOME])

# app = dash.Dash(__name__,external_stylesheets=[dbc.themes.BOOTSTRAP,'https://use.fontawesome.com/releases/v5.7.2/css/all.css'])

# excel_icon = html.I(className='far fa-file-excel', style=dict(display='inline-block'))
# btn_text=html.Div('Baixar parâmetros', style=dict(paddingRight='0.5vw',display='inline-block'))
# btn1_content = html.Span([btn_text,excel_icon])

gauge_size = "auto"
sidebar_size = 12
graph_size = 10
app.layout = dbc.Container(
    fluid = True,
    children = [
        cabecalho(app),
        dbc.Row(
            [
                dbc.Col(
                    [
                        dbc.Row(
                            dbc.Col(info_box, width = 12)
                        ),
                        dbc.Row(
                            dbc.Col(
                                get_variable, width = 12
                            )
                        ),
                        dbc.Row(
                            dbc.Col(
                                get_order, width = 12
                            )
                        ),

                        dbc.Row(
                            dbc.Col(
                                get_input, width = 12
                            )
                        )


                    ], width = 4
                ),
                dbc.Col([
                    dbc.Row(
                        dbc.Col(
                            dbc.Card(className='card secundary mb-3',
                                children =[
                                dbc.CardHeader(
                                    "Validação do modelo",
                                    style = {
                                        "text-align": "center",
                                        "color": "white",
                                        "border-radius":"1px",
                                        "border-width":"5px",
                                        "border-top":"1 px solid rgb(216, 216, 216)"
                                    }
                                ),
                                dbc.CardBody(
                                    [
                                    html.Div(children= dbc.Row([
                                                dbc.Col(dcc.Graph(id = 'plot1', figure = blank_fig()),width = 6),
                                                dbc.Col(dcc.Graph(id = 'plot2', figure = blank_fig()), width = 6)
                                                ], style = {"marginTop": 0}
                                        ),
                                    )   
                                    ]
                                ,style = {"marginTop": 0})
                                ])
                                ,width=12)
                                         
                            ,style = {"marginTop": 0}),

                    dbc.Row([
                            dbc.Col(get_prediction, width=12)
                            ], style = {"marginTop": 1}
                        ),
                    
                    dbc.Row([
                            dbc.Col(get_equation, width=12)
                            ], style = {"marginTop": 1}
                        )
                    ], width = 8
                )
            ], 
            style = {
                "marginTop": "2%"
            }
        ),

    dbc.Row( [
                            dbc.Col(
                                [html.H2(
                                    "Superfície gerada", style = {"marginTop": 10, "marginLeft":"10px","textAlign":"center"}
                                )], width = 12
                            )
                        ]),
    dbc.Row(
        [
        dbc.Col(width = 2),
        dbc.Col([
            dcc.Graph(id = 'surf-plot', figure = blank_fig())], width = 8
        ),
        dbc.Col(width = 2)], justify = 'center'

        ),
    dbc.Row(style = {"marginTop": 50})
    ]
)


def parse_contents(contents, filename):
    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
  
    if 'xlsx' in filename:
        df = pd.read_excel(BytesIO(decoded), engine='openpyxl')

        return df, html.Div([
            "Arquivo carregado com sucesso! :D."
        ])
     
    
    else:
        return pd.Datafeae(), html.Div([
            "Tipo de arquivo não suportado."
        ])


 

@app.callback(
    Output("modal", "is_open"),
    Output("output-data-upload", "children",allow_duplicate=True),
    Input("open", "n_clicks"),
    Input("close", "n_clicks"),
    State("modal", "is_open"),
    State("upload-data", "contents"),
    State("upload-data", "filename"),
    prevent_initial_call=True
)

def toggle_modal(n1, n2, is_open, contents, filename):

    if n1 or n2:
        return not is_open, None
    return is_open, None


# Primeiro dcc Store para almazenar o dataframe raw e gerar os options para os Dropdown menus (Store Callback)
@app.callback([Output('intermediate-value', 'data'),
                Output("output-data-upload", "children"),
                Output("x1-selector", "options"),
                Output("x2-selector", "options"),
                Output("x3-selector", "options")], 
                [Input("upload-data", "contents")], State("upload-data", "filename"),
                prevent_initial_call=True)
def update_output(contents, filename):
    if contents is not None:
        df, children = parse_contents(contents, filename)

    # some expensive data processing step by ex:
    # cleaned_df = slow_processing_step(value)
    tags = list(df.columns)

    # more generally, this line would be
    # json.dumps(cleaned_df)
    return df.to_json(date_format='iso', orient='split'),children,tags,tags,tags


# Atualizar o dropdown menu da variável de seleção baseado no options de x1-selector (Chained callback)
@app.callback([Output("var-rest","options")],
              [Input("x1-selector","options")],
              prevent_initial_call=True
              )
def update_constrains(x1):
    options = []
    if None in x1:
        options = ['Selecione primeiro todas as variáveis da análise :P']


    else:
        #options_ = ['Problema sem restrições']
        #options = x1.append('Problema sem restrições')
        
        options = x1
        options.append('Problema sem restrições') 
    return [options]


@app.callback([Output('intermediate-model-pls2','data'),Output('intermediate-model-pls3','data'),
               Output('intermediate-model-rmse','data'),Output('intermediate-model-C_json','data'),
               Output('intermediate-model-Y_pred','data'),Output('intermediate-model-classes','data')],
              [Input('botao-model-figs','n_clicks')],
              [State('intermediate-value', 'data'),State("x1-selector","value"), 
               State("x2-selector","value"), State("x3-selector","value"),
               State("order-selector","value"), State("var-rest","value"),State("var-value","value")],
               prevent_initial_call=True)
def reg_computation(n_clicks_calc,df_shared,x1,x2,x3,OR,REST_name,REST_value):
    # print(n_clicks_reg,x1,x2,x3,OR,REST_name,REST_value)
    # print('df++',df_shared)
    # input('df++')

    # if jsonified_data is None:
    #     raise PreventUpdate

    df_shared = pd.read_json(df_shared, orient='split')

    if not n_clicks_calc:
            figs_pngs_download = None
    else:

        tags = [x1,x2,x3]
        

        if None in tags:
            pass
        elif 'Problema sem restrições' in REST_name:
            tags2 = tags
            trans,NC = get_model(OR)

            RMSE,C = get_best_NC(df_shared[tags],trans,NC)
            RMSE2,C2 = get_best_NC(df_shared[tags2],trans,NC)

            Y_pred, pls2 = get_model_PLS(RMSE,df_shared[tags],C)
            pls3 = pls2


        
        else:
            tags2 = [x1,x2,REST_name]
            trans,NC = get_model(OR)

            RMSE,C = get_best_NC(df_shared[tags],trans,NC)
            RMSE2,C2 = get_best_NC(df_shared[tags2],trans,NC)
            
            Y_pred, pls2 = get_model_PLS(RMSE,df_shared[tags],C)
            Y_pred3, pls3 = get_model_PLS(RMSE2,df_shared[tags2],C2)

            # Indentation can be used for pretty-printing
            pls2_json = json.dumps(pls2.__dict__,
                                    allow_nan = True,
                                    indent = 6,cls=NpEncoder)
            pls3_json = json.dumps(pls3.__dict__,
                                    allow_nan = True,
                                    indent = 6,cls=NpEncoder)

            RMSE_json = json.dumps({"array": RMSE}, cls=NpEncoder)
            C_json = json.dumps({"array": C}, cls=NpEncoder)
            Y_pred_json = json.dumps({"array": Y_pred}, cls=NpEncoder)

            

            
            classes =[str(type(r)) for r in pls2.__dict__.values()]
            classes_json = json.dumps({"list": classes})
            # RMSE2_json = json.dumps(RMSE2,
            #                         allow_nan = True,
            #                         cls=NpEncoder)
            # C2_json = json.dumps(C2,
            #                         allow_nan = True,
            #                         cls=NpEncoder)
            # Y_pred2_json = json.dumps(Y_pred3,
            #                         allow_nan = True,
            #                         cls=NpEncoder)

    return pls2_json,pls3_json,RMSE_json,C_json,Y_pred_json,classes_json #,RMSE2_json,C2_json,Y_pred2_json)


@app.callback([Output("plot1", "figure"), Output("plot2", "figure"),Output("surf-plot", "figure")],
              [Input('intermediate-model-pls2','data'),Input('intermediate-model-pls3','data'),
               Input('intermediate-model-rmse','data'),Input('intermediate-model-C_json','data'),
               Input('intermediate-model-Y_pred','data'),Input('intermediate-model-classes','data')],
              [State('intermediate-value', 'data'),State("x1-selector","value"), 
               State("x2-selector","value"), State("x3-selector","value"),
               State("order-selector","value"),State("var-value","value")],
               prevent_initial_call=True)
def plotter_modelo(pls2_json,pls3_json,RMSE_json,C_json,Y_pred_json,classes_json,df_shared, x1,x2,x3, OR, REST_value):

    
    fig1 = blank_fig()
    fig2 = blank_fig()
    fig3 = blank_fig()
    tags = [x1,x2,x3]

    df_shared = pd.read_json(df_shared, orient='split')
    

    if None in tags:
        pass
    
    # loading json
    pls2_json = json.loads(pls2_json)
    pls3_json = json.loads(pls3_json)
    RMSE_json = json.loads(RMSE_json)
    C_json = json.loads(C_json)
    Y_pred_json = json.loads(Y_pred_json)
    classes = json.loads(classes_json)['list']


    # print(classes)
    # input('kd')

    # decoding json
    pls2_djson = NpDecoding(pls2_json, classes)
    pls3_djson = NpDecoding(pls3_json, classes)

    RMSE = np.asarray(RMSE_json["array"])
    C = np.asarray(C_json["array"])
    Y_pred = np.asarray(Y_pred_json["array"])


    # Define a new generic pls object
    pls2 = PLSRegression()
    pls3 = PLSRegression()

    # Assign all the dict keys of the loaded calibration object to the new pls object
    for k,v in enumerate(pls2_djson.keys()):
        pls2.__dict__[v] = pls2_djson[v]
        pls3.__dict__[v] = pls3_djson[v]

    fig1 = figure1(RMSE)
    fig2 = figure2(df_shared[tags],Y_pred)
    trans,NC = get_model(OR)
    Ccalc,Ccalc_r, var1, var2 = get_surface(df_shared[tags],trans,pls2, pls3,REST_value)
    fig3 = figure3(Ccalc,var1,var2,Ccalc_r,df_shared[tags])

    return fig1,fig2,fig3

@app.callback([Output('var-z-value','children'),Output('feasible_markdown', 'children')],
            [Input('botao-regressor','n_clicks')],
            [State('var-x1-value','value'),State('var-x2-value','value'),
            State('intermediate-model-pls3','data'),State('intermediate-model-classes','data'),
            State("x1-selector","value"),State("x2-selector","value"), State("x3-selector","value"),
            State("order-selector","value"), State("var-rest","value"),State("var-value","value")],
            prevent_initial_call=True)
def reg_point(n_clicks_point_reg,x1_ipt,x2_ipt,pls3_json,classes_json,x1,x2,x3,OR,REST_name,REST_value):
    markdown_criteria = dbc.Alert(
                                [html.H6("Previsão de viabilidade da região!", className="alert-heading"),
                                html.Hr(),
                                dcc.Markdown('''
                                    Calcule primeiro a predição para determinar a viabilidade do estado.
                                ''')], color="secondary",style={'marginTop':1,"marginBottom":1})
    
    markdown_value_z = dbc.Alert(str(None), color="secondary",style={'marginTop':1,"marginBottom":1})

    tags = [x1,x2,x3]



    if pls3_json is None:
        pass
    else:
        # loading json
        pls3_json = json.loads(pls3_json)
        classes = json.loads(classes_json)['list']

        # decoding json
        pls3_djson = NpDecoding(pls3_json, classes)

        # Define a new generic pls object
        pls3 = PLSRegression()

        # Assign all the dict keys of the loaded calibration object to the new pls object
        for k,v in enumerate(pls3_djson.keys()):
            pls3.__dict__[v] = pls3_djson[v]

    if OR is None:
        pass
    else:
        # Obtaining trans model
        trans,NC = get_model(OR)

    if None in tags:
        pass
    elif 'Problema sem restrições' in REST_name:
        # Predições
        if x1_ipt != None and x2_ipt != None:
            Chat=np.zeros((1,NC))
            Chat[0,:] = trans(x1_ipt,x2_ipt)
            y_hat = pls3.predict(Chat)
            y_hat= round(y_hat[0][0],2)


    

    else:
        # Predições
        if x1_ipt != None and x2_ipt != None:
            Chat=np.zeros((1,NC))
            Chat[0,:] = trans(x1_ipt,x2_ipt)
            y_hat = pls3.predict(Chat)
            y_hat= round(y_hat[0][0],2)

            if y_hat != None and y_hat >= REST_value:
                # markdown_criteria = dcc.Markdown('''
                #                     ###### Predição da viabilidade da região:
                #                     O estado estimado `z` corresponde a um estado **víavel**.
                #                 ''')
                markdown_criteria = dbc.Alert(
                    [html.H6("Previsão de viabilidade da região!", className="alert-heading"),
                        html.Hr(),
                        dcc.Markdown('''
                                    O estado estimado **z** corresponde a um estado **víavel**.
                                ''')], color="success",style={'marginTop':1,"marginBottom":1})
                markdown_value_z = dbc.Alert(str(y_hat), color="success",style={'marginTop':1,"marginBottom":1})

            else:
                
                # markdown_criteria=dcc.Markdown('''
                #                     ###### Predição da viabilidade da região:
                #                     O estado estimado `z` corresponde a um estado ** não víavel**.
                #                 ''')
                markdown_criteria = dbc.Alert(
                    [html.H6("Previsão de viabilidade da região!", className="alert-heading"),
                        html.Hr(),
                        dcc.Markdown('''
                                    O estado estimado **z** corresponde a um estado **não víavel**.
                                ''')], color="danger",style={'marginTop':1,"marginBottom":1})
                markdown_value_z = dbc.Alert(str(y_hat), color="danger",style={'marginTop':1,"marginBottom":1})

    return markdown_value_z,markdown_criteria


@app.callback(
    Output("download-xls", "data"),
    [Input('parameters-download-btn','n_clicks')],
    [State('intermediate-model-pls3','data'),State('intermediate-model-classes','data')])
def download_parameters(n_clicks_parameters,pls2_json,classes_json):

    # data_xls_download = None
    # parametros_xls = pd.DataFrame()


    if n_clicks_parameters>0:
        # loading json
        pls2_json = json.loads(pls2_json)
        classes = json.loads(classes_json)['list']

        # decoding json
        pls2_djson = NpDecoding(pls2_json, classes)

        # Define a new generic pls object
        pls2 = PLSRegression()

        # Assign all the dict keys of the loaded calibration object to the new pls object
        for k,v in enumerate(pls2_djson.keys()):
            pls2.__dict__[v] = pls2_djson[v]

        parametros_xls = pd.DataFrame(data=np.column_stack((pls2._x_mean,pls2._x_std,pls2.coef_[0],np.ones(len(pls2._x_mean))*np.nan)),columns=['mean','std','coef','intercept'])
        parametros_xls.iloc[0]['intercept'] = pls2.intercept_


        data_xls_download = dcc.send_data_frame(parametros_xls.to_excel, "Datafile.xlsx", sheet_name="Sheet_name_1")

        return data_xls_download

        # # Create a Pandas Excel writer using XlsxWriter as the engine.
        # writer = pd.ExcelWriter('Datafile.xlsx', engine='xlsxwriter')
        # parametros_xls.to_excel(writer, sheet_name='parameters-PLS')
        # # Close the Pandas Excel writer and output the Excel file.
        # writer.close()
        # data_xls_download = dcc.send_file('Datafile.xlsx')

        # return data_xls_download
    
    else:
        
        data_xls_download = None
        # raise  PreventUpdate
        return dash.no_update

@app.callback(Output("download-pngs", "data"), 
[Input('figures-download-btn','n_clicks')],
[State("plot1", "figure"), State("plot2", "figure"),State("surf-plot", "figure")])
def download_figures(n_clicks_figs,fig1_,fig2_,fig3_):

    if n_clicks_figs >0:
        fig1 = go.Figure(fig1_)
        fig2 = go.Figure(fig2_)
        fig3 = go.Figure(fig3_)
        count = 0
        # Crie um arquivo ZIP para conter as imagens
        output = io.BytesIO()
        with zipfile.ZipFile(output, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for selected_image in [fig1,fig2,fig3]:
                selected_image.update_layout(template = "Journal")
                count = count +1
                img_bytes = selected_image.to_image(format="png")
                filename = 'fig'+str(count)+'.png'
                zipf.writestr(filename, img_bytes)
                selected_image.update_layout(template = "superhero")

        output.seek(0)
        figs_pngs_download = dcc.send_bytes(output.read(), filename="imagens.zip")

        return figs_pngs_download

    else:
        raise PreventUpdate


            











# @app.callback([Output("plot1", "figure"), Output("plot2", "figure"),Output("surf-plot", "figure"),
#                Output('var-z-value','children'),Output('feasible_markdown', 'children'),Output("download-xls", "data"),
#                Output("download-pngs", "data")],
#             #    , , Output("surf-fig", "figure")],
#               [Input("botao-carregar", "n_clicks"),
#               Input("upload-data", "contents"), Input("upload-data", "filename"), 
#               Input('botao-regressor','n_clicks'), Input('parameters-download-btn','n_clicks'),Input('figures-download-btn','n_clicks')], 
#               [State("x1-selector","value"), State("x2-selector","value"), State("x3-selector","value"),
#                State("order-selector","value"), State("var-rest","value"),State("var-value","value"),
#                State('var-x1-value','value'),State('var-x2-value','value')],
#                 prevent_initial_call=True
#               )

# def plots_modelo(n_clicks, contents, filename,n_clicks_reg,n_clicks_parameters,n_clicks_figs, x1,x2,x3,OR,REST_name,REST_value,x1_ipt,x2_ipt):
    

#     fig1 = blank_fig()
#     fig2 = blank_fig()
#     fig3 = blank_fig()
#     y_hat = None

#     markdown_criteria = dbc.Alert(
#                                 [html.H6("Previsão de viabilidade da região!", className="alert-heading"),
#                                 html.Hr(),
#                                 dcc.Markdown('''
#                                     Calcule primeiro a predição para determinar a viabilidade do estado.
#                                 ''')], color="secondary",style={'marginTop':1,"marginBottom":1})
    
#     markdown_value_z = dbc.Alert(str(None), color="secondary",style={'marginTop':1,"marginBottom":1})

#     data_xls_download = None
#     figs_pngs_download = None


#     if contents is not None:
#         df, children = parse_contents(contents, filename)
#     tags = [x1,x2,x3]
    

#     if None in tags:
#         pass
#     elif 'Problema sem restrições' in REST_name:
#         tags2 = tags
#         trans,NC = get_model(OR)
#         RMSE,C = get_best_NC(df[tags],trans,NC)

#         fig1 = figure1(RMSE)

#         RMSE2,C2 = get_best_NC(df[tags2],trans,NC)
#         Y_pred, pls2 = get_model_PLS(RMSE,df[tags2],C)

#         fig2 = figure2(df[tags],Y_pred)

#         pls3 = pls2

#         # Predições
#         if x1_ipt != None and x2_ipt != None:
#             Chat=np.zeros((1,NC))
#             Chat[0,:] = trans(x1_ipt,x2_ipt)
#             y_hat = pls3.predict(Chat)
#             y_hat= round(y_hat[0][0],2)

#         Ccalc,Ccalc_r, var1, var2 = get_surface(df[tags],trans,pls2, pls3,REST_value)
#         fig3 = figure3(Ccalc,var1,var2,Ccalc_r,df[tags])

#         parametros_xls = pd.DataFrame(data=np.column_stack((pls2._x_mean,pls2._x_std,pls2.coef_[0],np.ones(len(pls2._x_mean))*np.nan)),columns=['mean','std','coef','intercept'])
#         parametros_xls.iloc[0]['intercept'] = pls2.intercept_
#         if not n_clicks_parameters:
#             data_xls_download = None
#             # raise PreventUpdate
#         else:
#             # Create a Pandas Excel writer using XlsxWriter as the engine.
#             writer = pd.ExcelWriter('Datafile.xlsx', engine='xlsxwriter')
#             parametros_xls.to_excel(writer, sheet_name='parameters-PLS')
#             # Close the Pandas Excel writer and output the Excel file.
#             writer.close()
#             data_xls_download = dcc.send_file('Datafile.xlsx')

#         if not n_clicks_figs:
#             figs_pngs_download = None
#         else:
#             count =0
#             # Crie um arquivo ZIP para conter as imagens
#             output = io.BytesIO()
#             with zipfile.ZipFile(output, 'w', zipfile.ZIP_DEFLATED) as zipf:
#                 for selected_image in [fig1,fig2,fig3]:
#                     selected_image.update_layout(template = "Journal")
#                     count = count +1
#                     img_bytes = selected_image.to_image(format="png")
#                     filename = 'fig'+str(count)+'.png'
#                     zipf.writestr(filename, img_bytes)
#                     selected_image.update_layout(template = "superhero")

#             output.seek(0)
#             figs_pngs_download = dcc.send_bytes(output.read(), filename="imagens.zip")

#     else:
#         tags2 = [x1,x2,REST_name]
#         trans,NC = get_model(OR)
#         RMSE,C = get_best_NC(df[tags],trans,NC)

#         fig1 = figure1(RMSE)

#         RMSE2,C2 = get_best_NC(df[tags2],trans,NC)
        
        

#         Y_pred, pls2 = get_model_PLS(RMSE,df[tags],C)

#         fig2 = figure2(df[tags],Y_pred)

#         Y_pred3, pls3 = get_model_PLS(RMSE2,df[tags2],C2)

#         # Predições
#         if x1_ipt != None and x2_ipt != None:
#             Chat=np.zeros((1,NC))
#             Chat[0,:] = trans(x1_ipt,x2_ipt)
#             y_hat = pls3.predict(Chat)
#             y_hat= round(y_hat[0][0],2)


#         if y_hat != None and y_hat >= REST_value:
#             # markdown_criteria = dcc.Markdown('''
#             #                     ###### Predição da viabilidade da região:
#             #                     O estado estimado `z` corresponde a um estado **víavel**.
#             #                 ''')
#             markdown_criteria = dbc.Alert(
#                 [html.H6("Previsão de viabilidade da região!", className="alert-heading"),
#                     html.Hr(),
#                     dcc.Markdown('''
#                                 O estado estimado `z` corresponde a um estado **víavel**.
#                             ''')], color="success",style={'marginTop':1,"marginBottom":1})
#             markdown_value_z = dbc.Alert(str(y_hat), color="success",style={'marginTop':1,"marginBottom":1})

#             Ccalc,Ccalc_r, var1, var2 = get_surface(df[tags],trans,pls2, pls3,REST_value)
    
#         else:
            
#             # markdown_criteria=dcc.Markdown('''
#             #                     ###### Predição da viabilidade da região:
#             #                     O estado estimado `z` corresponde a um estado ** não víavel**.
#             #                 ''')
#             markdown_criteria = dbc.Alert(
#                 [html.H6("Previsão de viabilidade da região!", className="alert-heading"),
#                     html.Hr(),
#                     dcc.Markdown('''
#                                 O estado estimado `z` corresponde a um estado **não víavel**.
#                             ''')], color="danger",style={'marginTop':1,"marginBottom":1})
#             markdown_value_z = dbc.Alert(str(y_hat), color="danger",style={'marginTop':1,"marginBottom":1})
            


#         # f"This report covers the time period spanning {start_date} to {end_date}"

#         Ccalc,Ccalc_r, var1, var2 = get_surface(df[tags],trans,pls2, pls3,REST_value)
#         # print(tags)

#         fig3 = figure3(Ccalc,var1,var2,Ccalc_r,df[tags])

#         parametros_xls = pd.DataFrame(data=np.column_stack((pls2._x_mean,pls2._x_std,pls2.coef_[0],np.ones(len(pls2._x_mean))*np.nan)),columns=['mean','std','coef','intercept'])
#         parametros_xls.iloc[0]['intercept'] = pls2.intercept_
#         if not n_clicks_parameters:
#             data_xls_download = None
#             # raise PreventUpdate
#         else:
#             # Create a Pandas Excel writer using XlsxWriter as the engine.
#             writer = pd.ExcelWriter('Datafile.xlsx', engine='xlsxwriter')
#             parametros_xls.to_excel(writer, sheet_name='parameters-PLS')
#             # Close the Pandas Excel writer and output the Excel file.
#             writer.close()
#             data_xls_download = dcc.send_file('Datafile.xlsx')

#         if not n_clicks_figs:
#             figs_pngs_download = None
#         else:
#             count =0
#             # Crie um arquivo ZIP para conter as imagens
#             output = io.BytesIO()
#             with zipfile.ZipFile(output, 'w', zipfile.ZIP_DEFLATED) as zipf:
#                 for selected_image in [fig1,fig2,fig3]:
#                     selected_image.update_layout(template = "Journal")
#                     count = count +1
#                     img_bytes = selected_image.to_image(format="png")
#                     filename = 'fig'+str(count)+'.png'
#                     zipf.writestr(filename, img_bytes)
#                     selected_image.update_layout(template = "superhero")

#             output.seek(0)
#             figs_pngs_download = dcc.send_bytes(output.read(), filename="imagens.zip")




#     return (fig1, fig2, fig3,markdown_value_z,markdown_criteria,data_xls_download,figs_pngs_download)#, fig2, fig3





#---------------------
# @app.callback([Output("oi-id","children")],
#               [Input("botao-carregar", "n_clicks"),
#               Input("upload-data", "contents"), Input("upload-data", "filename")], [State("x1-selector","value"), State("x2-selector","value"), State("x3-selector","value"),
#                State("order-selector","value"), State("var-rest","value"),State("var-value","value")],
#                 prevent_initial_call=True
#               )


# def plots_modelo(n_clicks, contents, filename, x1,x2,x3,OR,REST_name,REST_value):
#     print(type(OR))
#     if contents is not None:
#         df, children = parse_contents(contents, filename)
#     tags = [x1,x2,x3]

#     if None in tags:
#         pass
#     else:
#         dados_filter = copy.deepcopy(df[tags])
#         dados_filter = dados_filter[dados_filter[REST_name]>REST_value]
    



   

#     return [html.H1("Hello bonitos")]



if __name__ == '__main__':
    app.run_server(
        port=8050,
        host='0.0.0.0',
        debug=False)  
