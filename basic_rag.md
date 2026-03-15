<a id='1'></a>
## 1 - Introduction

---

<a id='1-1'></a>
### 1.1 RAG architecture overview

As presented in the lectures, a simplified diagram for RAG is as follows:

<div align="center">
  <img src="images/rag_overview.png" alt="RAG Overview" width="60%">
</div>

This assignment is designed to guide you through the general workflow. You will employ a pre-implemented retriever to obtain relevant data for a given query, format this data, and create a new prompt that includes both the query and the retrieved information. Finally, you will have the opportunity to compare the results of queries both with and without the RAG system to assess its impact on the LLM's response. 

Alright, let's roll up our sleeves and get started!

<a id='1-2'></a>
### 1.2 Importing the necessary libraries

Run the cell below to import the necessary libraries for this assignment.


```python
from utils import (
    retrieve, 
    pprint, 
    generate_with_single_input, 
    read_dataframe, 
    display_widget
)
import unittests
```

    Warning: You are sending unauthenticated requests to the HF Hub. Please set a HF_TOKEN to enable higher rate limits and faster downloads.
    Could not cache non-existence of file. Will ignore error and continue. Error: [Errno 30] Read-only file system: '/home/jovyan/models/models--BAAI--bge-base-en-v1.5/.no_exist/a5beb1e3e68b9ab74eb54cfd186867f64f240e1a/adapter_config.json'



    Loading weights:   0%|          | 0/199 [00:00<?, ?it/s]


    [1mBertModel LOAD REPORT[0m from: BAAI/bge-base-en-v1.5
    Key                     | Status     |  | 
    ------------------------+------------+--+-
    embeddings.position_ids | UNEXPECTED |  | 
    
    [3mNotes:
    - UNEXPECTED[3m	:can be ignored when loading from different task/architecture; not ok if you expect identical arch.[0m


<a id='2'></a>

<a id='2'></a>
## 2 - Loading the dataset

---

In this assignment, you’ll work with the Kaggle dataset [News Headlines 2024](https://www.kaggle.com/datasets/dylanjcastillo/news-headlines-2024). This dataset contains thousands of news headlines and related information from BBC News.



```python
NEWS_DATA = read_dataframe("news_data_dedup.csv")
```

Let's check the data structure.


```python
pprint(NEWS_DATA[9:11])
```

    [
      {
        "guid": "5dae28f191cfd1047f67c409e616fc3f",
        "title": "Paris's Moulin Rouge loses windmill sails overnight",
        "description": "The cause of the sails' collapse from the roof of the world famous cabaret club is not yet clear.",
        "venue": "BBC",
        "url": "https://www.bbc.co.uk/news/world-europe-68895836",
        "published_at": "2024-04-25",
        "updated_at": "2024-04-26"
      },
      {
        "guid": "d2c3ff79d4e068911d05416ca061cd51",
        "title": "Ukraine uses longer-range US missiles for first time",
        "description": "Missiles secretly delivered this month have been used to strike Russian targets in Crimea, US media say.",
        "venue": "BBC",
        "url": "https://www.bbc.co.uk/news/world-europe-68893196",
        "published_at": "2024-04-25",
        "updated_at": "2024-04-26"
      }
    ]


Important fields are `title`, `description`, `url` and `published_at`. These fields will give good information to the LLM to answer the majority of questions with good enough data.

<a id='3'></a>
## 3 - Main Functions
---
Two functions will be provided to you:
- `query_news`: Given a list of indices, this function returns all documents corresponding to those indices.
- `retrieve`: Given a query and an integer called `top_k`, this function retrieves the `top_k` most relevant documents.

The functions you will implement are:
- `get_relevant_data`: This function takes a query and a `top_k` value and returns the `top_k` relevant documents.
- `format_relevant_data`: Given a list of documents, this function creates a formatted string with the document information.

You will then use these functions to create your own small RAG pipeline and see it in action!

<a id='3-1'></a>
### 3.1 Query news by index function

This simple function just simplifies the return of documents given a list of indices. It is given to you.


```python
def query_news(indices):
    """
    Retrieves elements from a dataset based on specified indices.

    Parameters:
    indices (list of int): A list containing the indices of the desired elements in the dataset.
    
    Returns:
    list: A list of elements from the dataset corresponding to the indices provided in list_of_indices.
    """
     
    output = [NEWS_DATA[index] for index in indices]

    return output
```


```python
# Fetching some indices
indices = [3, 6, 9]
pprint(query_news(indices))
```

    [
      {
        "guid": "e696224ac208878a5cec8bdc9f97c632",
        "title": "Europe risks dying and faces big decisions - Macron",
        "description": "The French president delivers a stark warning for Europe to act fast to survive in a changing world.",
        "venue": "BBC",
        "url": "https://www.bbc.co.uk/news/world-europe-68898887",
        "published_at": "2024-04-25",
        "updated_at": "2024-04-26"
      },
      {
        "guid": "4f585bad8f61b715fbafe2f022ab0ae8",
        "title": "Supreme Court divided on whether Trump has immunity",
        "description": "The justices discussed immunity, coups, pardons, Operation Mongoose - and the future of democracy.",
        "venue": "BBC",
        "url": "https://www.bbc.co.uk/news/world-us-canada-68901817",
        "published_at": "2024-04-25",
        "updated_at": "2024-04-26"
      },
      {
        "guid": "5dae28f191cfd1047f67c409e616fc3f",
        "title": "Paris's Moulin Rouge loses windmill sails overnight",
        "description": "The cause of the sails' collapse from the roof of the world famous cabaret club is not yet clear.",
        "venue": "BBC",
        "url": "https://www.bbc.co.uk/news/world-europe-68895836",
        "published_at": "2024-04-25",
        "updated_at": "2024-04-26"
      }
    ]


<a id='3-2'></a>
### 3.2 Retrieve function

The `retrieve` function is an essential part of our RAG system. While the full code is provided in the `utils.py` file, and you can examine it there, you will focus on understanding its input parameters and output for now. In Module 2, you will explore the detailed workings and various techniques for document retrieval.

**Parameters:**
- `query`: A string representing the search query for which you want to find the most relevant documents.
- `top_k`: An integer indicating the number of top similar documents to return.

**Output:**
- The function returns a list of indices corresponding to the top `k` most similar documents from the corpus, based on their similarity scores with the query.

**Call:**

```Python
retrieve(query: str, top_k: int)
```

As you move forward in this course, you'll gain deeper insights into how this function utilizes embeddings and similarity measures to perform effective document retrieval.


```python
# Let's test the retrieve function!
indices = retrieve("Concerts in North America", top_k = 1)
print(indices)
```

    [350]



```python
# Now let's query the corresponding news_
retrieved_documents = query_news(indices)
pprint(retrieved_documents)
```

    [
      {
        "guid": "927257674585bb6ef669cf2c2f409fa7",
        "title": "\u2018The working class can\u2019t afford it\u2019: the shocking truth about the money bands make on tour",
        "description": "As Taylor Swift tops $1bn in tour revenue, musicians playing smaller venues are facing pitiful fees and frequent losses. Should the state step in to save our live music scene?When you see a band playing to thousands of fans in a sun-drenched festival field, signing a record deal with a major label or playing endlessly from the airwaves, it\u2019s easy to conjure an image of success that comes with some serious cash to boot \u2013 particularly when Taylor Swift has broken $1bn in revenue for her current Eras tour. But looks can be deceiving. \u201cI don\u2019t blame the public for seeing a band playing to 2,000 people and thinking they\u2019re minted,\u201d says artist manager Dan Potts. \u201cBut the reality is quite different.\u201dPost-Covid there has been significant focus on grassroots music venues as they struggle to stay open. There\u2019s been less focus on the actual ability of artists to tour these venues. David Martin, chief executive officer of the Featured Artists Coalition (FAC), says we\u2019re in a \u201ccost-of-touring crisis\u201d. Pretty much every cost attached to touring \u2013 van hire, crew, travel, accommodation, food and drink \u2013 has gone up, while fees and audiences often have not. \u201c[Playing] live is becoming financially unsustainable for many artists,\u201d he says. \u201cArtists are seeing [playing] live as a loss leader now. That\u2019s if they can even afford to make it work in the first place.\u201d Continue reading...",
        "venue": "The Guardian",
        "url": "https://www.theguardian.com/music/2024/apr/25/shocking-truth-money-bands-make-on-tour-taylor-swift",
        "published_at": "2024-04-25",
        "updated_at": "2024-04-26"
      }
    ]


<a id='3-3'></a>
### 3.3 Get relevant data

<a id='ex01'></a>
### Exercise 1

In this exercise, you will create a function that inputs a `query` a `top_k` and returns a list with the `relevant items`. You will consolidate the two functions described so far into only one.

<details>
<summary style="color: green;">Hint 1</summary>
The function to get the relevant indices is the <code>retrieve</code> function. Remember that it inputs a <code>query</code> and the <code>top_k</code> and returns a <b>list of indices</b>.
</details>
<details>
<summary style="color: green;">Hint 2</summary>
The function to get the relevant data is the <code>query_news</code> function. It inputs a <b>set of indices</b> and outputs a list with the relevant data.
</details>


```python
 

def get_relevant_data(query: str, top_k: int = 5) -> list[dict]:
    """
    Retrieve and return the top relevant data items based on a given query.

    This function performs the following steps:
    1. Retrieves the indices of the top 'k' relevant items from a dataset based on the provided `query`.
    2. Fetches the corresponding data for these indices from the dataset.

    Parameters:
    - query (str): The search query string used to find relevant items.
    - top_k (int, optional): The number of top items to retrieve. Default is 5.

    Returns:
    - list[dict]: A list of dictionaries containing the data associated 
      with the top relevant items.

    """
    ### START CODE HERE ###

    # Retrieve the indices of the top_k relevant items given the query
    relevant_indices = retrieve(query,top_k)

    # Obtain the data related to the items using the indices from the previous step
    relevant_data = query_news(relevant_indices)

    ### END CODE HERE
    
    return relevant_data
```


```python
query = "Greatest storms in the US"
relevant_data = get_relevant_data(query, top_k = 1)
pprint(relevant_data)
```

    [
      {
        "guid": "3ca548fe82c3fcae2c4c0c635d03eb2e",
        "title": "Large tornado seen touching down in Nebraska",
        "description": "Severe and powerful storms have moved across several US states, leaving many experiencing power shortages.",
        "venue": "BBC",
        "url": "https://www.bbc.co.uk/news/world-us-canada-68860070",
        "published_at": "2024-04-26",
        "updated_at": "2024-04-28"
      }
    ]


**Expected output**
```
[{'guid': '3ca548fe82c3fcae2c4c0c635d03eb2e',
  'title': 'Large tornado seen touching down in Nebraska',
  'description': 'Severe and powerful storms have moved across several US '
                 'states, leaving many experiencing power shortages.',
  'venue': 'BBC',
  'url': 'https://www.bbc.co.uk/news/world-us-canada-68860070',
  'published_at': '2024-04-26',
  'updated_at': '2024-04-28'}]
```


```python
# Run this cell to perform several tests on your function. If you receive "All test passed!" it means that your solution will likely pass the autograder too.
unittests.test_get_relevant_data(get_relevant_data)
```

    [92m All tests passed!


<a id='3-4'></a>

<a id='3-4'></a>
### 3.4 Formatting the relevant rata


<a id='ex02'></a>

<a id='ex02'></a>
### Exercise 2

In this exercise, you will create the `format_relevant_data` function that takes a list of data and formats it.

**Note:** You can adjust the layout, but your output **must** include these pieces of information:

* News title
* News description
* News published date
* News URL

You must include the exact keywords `'title'`, `'url'`, `'published_at'`, and `'description'` somewhere in your output (uppercase or lowercase versions are fine). These are required for grading.

**Tip:** It’s recommended to use **double quotes** for strings and **single quotes** for dictionary keys (for example, `data['title']`).
Here’s one way you could format it:

```python
f"""
Title: {news_1_title}, Description: {news_1_description}, Published at: {news_1_published_date}\nURL: {news_1_URL}
Title: {news_2_title}, Description: {news_2_description}, Published at: {news_2_published_date}\nURL: {news_2_URL}
...
Title: {news_k_title}, Description: {news_k_description}, Published at: {news_k_published_date}\nURL: {news_k_URL}
"""
```

<details>  
<summary style="color: green;">Hint 1</summary>  
<p>You can access each property of a document using its key in the dictionary.</p>  
<p>For example, to get the title: <code>document['title']</code></p>  
<p>You can format it like this: <code>f"Title: {document['title']}"</code></p>  
</details>

<details>  
<summary style="color: green;">Hint 2</summary>  
<p>Remember to append each formatted document to the <code>formatted_documents</code> list by doing <code>formatted_documents.append(formatted_document)</code>.</p>  
</details>


```python


def format_relevant_data(relevant_data):
    """
    Formats a list of relevant documents into a structured string for use in a RAG system.

    Parameters:
    relevant_data (list): A list with relevant data.

    Returns:
    str: A formatted string with the relevant documents, structured for use in a Retrieval-Augmented Generation (RAG) system."
    """

    ### START CODE HERE ###

    # Create a list to store the formatted documents
    formatted_documents = []
    
    # Iterates over each relevant document.
    for document in relevant_data:

        # Formats each document into a structured layout string. Remember that each document is in one different line. So you should add a new line character after each document added.
        formatted_document = f"""
Title: {document['title']}, Description: {document['description']}, Published at: {document['published_at']}\nURL: {document['url']}
"""

        # Append the formatted document string to the formatted_documents list
        formatted_documents.append(formatted_document)
    
    ### END CODE HERE ###
    
    # Returns the final augmented prompt string.

    return "\n".join(formatted_documents)
```


```python
example_data = NEWS_DATA[4:8]
```

Now let's test your function with some queries.


```python
print(format_relevant_data(example_data))
```

    
    Title: Prosecutors ask for halt to case against Spain PM's wife, Description: Pedro Sánchez is deciding whether to resign after a case against his wife by an anti-corruption group., Published at: 2024-04-25
    URL: https://www.bbc.co.uk/news/world-europe-68895727
    
    
    Title: WATCH: Would you pay a tourist fee to enter Venice?, Description: From Thursday visitors making a trip to the famous city at peak times will be charged a trial entrance fee., Published at: 2024-04-25
    URL: https://www.bbc.co.uk/news/world-europe-68898441
    
    
    Title: Supreme Court divided on whether Trump has immunity, Description: The justices discussed immunity, coups, pardons, Operation Mongoose - and the future of democracy., Published at: 2024-04-25
    URL: https://www.bbc.co.uk/news/world-us-canada-68901817
    
    
    Title: More than 150 killed as heavy rains pound Tanzania, Description: The prime minister warns that El Niño-triggered heavy rains are likely to continue into May., Published at: 2024-04-25
    URL: https://www.bbc.co.uk/news/world-africa-68896454
    



```python
# Test your function!
unittests.test_format_relevant_data(format_relevant_data)
```

    [92m All tests passed!


<a id='3-5'></a>
### 3.5 Generate the final prompt

The next function is given to you. It will generate the final prompt, integrating it with the query. Feel free to change the prompt and experiment how different prompts impact the final result!


```python

def generate_final_prompt(query, top_k=5, use_rag=True, prompt=None):
    """
    Generates a final prompt based on a user query, optionally incorporating relevant data using retrieval-augmented generation (RAG).

    Args:
        query (str): The user query for which the prompt is to be generated.
        top_k (int, optional): The number of top relevant data pieces to retrieve and incorporate. Default is 5.
        use_rag (bool, optional): A flag indicating whether to use retrieval-augmented generation (RAG)
                                  by including relevant data in the prompt. Default is True.
        prompt (str, optional): A template string for the prompt. It can contain placeholders {query} and {documents}
                                for formatting with the query and formatted relevant data, respectively.

    Returns:
        str: The generated prompt, either consisting solely of the query or expanded with relevant data
             formatted for additional context.
    """
    # If RAG is not being used, format the prompt with just the query or return the query directly
    if not use_rag:
        return query

    # Retrieve the top_k relevant data pieces based on the query
    relevant_data = get_relevant_data(query, top_k=top_k)

    # Format the retrieved relevant data
    retrieve_data_formatted = format_relevant_data(relevant_data)

    # If no custom prompt is provided, use the default prompt template
    if prompt is None:
        prompt = (
            f"Answer the user query below. There will be provided additional information for you to compose your answer. "
            f"The relevant information provided is from 2024 and it should be added as your overall knowledge to answer the query, "
            f"you should not rely only on this information to answer the query, but add it to your overall knowledge."
            f"Query: {query}\n"
            f"2024 News: {retrieve_data_formatted}"
        )
    else:
        # If a custom prompt is provided, format it with the query and formatted relevant data
        prompt = prompt.format(query=query, documents=retrieve_data_formatted)

    return prompt
```


```python
print(generate_final_prompt("Tell me about the US GDP in the past 3 years."))
```

    Answer the user query below. There will be provided additional information for you to compose your answer. The relevant information provided is from 2024 and it should be added as your overall knowledge to answer the query, you should not rely only on this information to answer the query, but add it to your overall knowledge.Query: Tell me about the US GDP in the past 3 years.
    2024 News: 
    Title: America's Economy Is No. 1. That Means Trouble, Description: If you want a single number to capture America’s economic stature, here it is: This year, the U.S. will account for 26.3% of the global gross domestic product, the highest in almost two decades. That’s based on the latest projections from the International Monetary Fund. According to the IMF, Europe’s share of world GDP has dropped 1.4 percentage points since 2018, and Japan’s by 2.1 points. The U.S. share, by contrast, is up 2.3 points., Published at: 2024-04-26
    URL: https://www.wsj.com/articles/americas-economy-is-no-1-that-means-trouble-d008e4bd
    
    
    Title: Do the GDP and Dow Reflect American Well-Being?, Description: Do the GDP and Dow Reflect American Well-Being?, Published at: 2024-04-25
    URL: https://www.wsj.com/economy/gdp-and-the-dow-are-up-but-what-about-american-well-being-87f90e6d?mod=wknd_pos1
    
    
    Title: America's Economy Is No. 1. That Means Trouble., Description: Solid growth, big deficits and a strong dollar stir memories of past crises., Published at: 2024-04-25
    URL: https://www.wsj.com/articles/us-economy-strongest-world-imf-projections-8e707514
    
    
    Title: Live Markets: Stock Futures Fall, Yields Jump After GDP Report, Description: New data showed U.S. economic growth slowed in the first quarter. Gross domestic product expanded at a 1.6% seasonally- and inflation-adjusted annual rate in the first quarter., Published at: 2024-04-25
    URL: https://www.wsj.com/articles/live-markets-updates-c5186430
    
    
    Title: GDP and the Dow Are Up. But What About American Well-Being?, Description: The standard ways of measuring economic growth don’t capture what life is like for real people. A new metric offers a better alternative, especially for seeing disparities across the country., Published at: 2024-04-25
    URL: https://www.wsj.com/articles/gdp-and-the-dow-are-up-but-what-about-american-well-being-87f90e6d
    


<a id='3-6'></a>
### 3.6 LLM call

Now let's integrate the function above to feed an LLM. Its parameters are:

- `query`: the query to be passed to the LLM.
- `use_rag`: a boolean telling whether using RAG or not. This parameter will help you compare queries using a RAG system and not using it.
- `model`: the model to be used. You might change it, but the standard is the Llama 3 Billion parameter.  


```python
def llm_call(query, top_k = 5, use_rag = True, prompt = None):
    """
    Calls the LLM to generate a response based on a query, optionally using retrieval-augmented generation.

    Args:
        query (str): The user query that will be processed by the language model.
        use_rag (bool, optional): A flag that indicates whether to use retrieval-augmented generation by 
                                  incorporating relevant documents into the prompt. Default is True.

    Returns:
        str: The content of the response generated by the language model.
    """
    

    # Get the prompt with the query + relevant documents
    prompt = generate_final_prompt(query, top_k, use_rag, prompt)

    # Call the LLM
    generated_response = generate_with_single_input(prompt)

    # Get the content
    generated_message = generated_response['content']
    
    return generated_message
```


```python
query = "Tell me about the US GDP in the past 3 years."
```


```python
print(llm_call(query, use_rag = True))
```

    Based on the provided information from 2024, I can give you an overview of the US GDP in the past 3 years.
    
    According to the International Monetary Fund (IMF), the US GDP share of the global gross domestic product has been increasing. In 2024, the US is projected to account for 26.3% of the global GDP, the highest in almost two decades. This is a 2.3 percentage point increase since 2018.
    
    In terms of actual GDP growth, the US economy experienced solid growth in 2024, but the growth slowed in the first quarter of 2024. The GDP expanded at a 1.6% seasonally- and inflation-adjusted annual rate in the first quarter of 2024.
    
    Here's a brief summary of the US GDP growth in the past 3 years:
    
    - 2022: The US GDP growth was strong, but the data is not provided in the given information.
    - 2023: The US GDP growth was solid, but the exact rate is not provided in the given information.
    - 2024 (first quarter): The US GDP growth slowed, expanding at a 1.6% seasonally- and inflation-adjusted annual rate.
    
    Please note that the provided information is from 2024, and you should consider adding it to your overall knowledge to answer the query accurately.



```python
print(llm_call(query, use_rag = False))
```

    The US GDP (Gross Domestic Product) is a widely used indicator of a country's economic performance. Here's a brief overview of the US GDP for the past 3 years (2020-2022):
    
    1. **2020**: The US GDP experienced a significant contraction due to the COVID-19 pandemic. According to the Bureau of Economic Analysis (BEA), the US GDP declined by 3.4% in 2020, with a quarterly GDP growth rate of -31.4% in Q2 2020 (the largest decline on record). However, the economy rebounded in the second half of the year, with a quarterly GDP growth rate of 33.4% in Q3 2020 and 6.7% in Q4 2020.
    
    2. **2021**: The US GDP grew by 5.7% in 2021, with a quarterly GDP growth rate of 1.6% in Q1 2021, 6.9% in Q2 2021, 2.6% in Q3 2021, and 2.3% in Q4 2021. The economy continued to recover from the pandemic, driven by government stimulus, vaccination efforts, and a rebound in consumer spending.
    
    3. **2022**: The US GDP grew by 2.1% in 2022, with a quarterly GDP growth rate of 1.6% in Q1 2022, 2.6% in Q2 2022, 2.0% in Q3 2022, and 2.1% in Q4 2022. The economy faced headwinds from inflation, supply chain disruptions, and the ongoing pandemic, but continued to grow at a moderate pace.
    
    It's worth noting that these figures are subject to revision and may not reflect the most up-to-date numbers. Additionally, the impact of the pandemic and other global events on the US economy is still being felt, and the future trajectory of the economy is uncertain.
    
    Here are the actual GDP numbers for the past 3 years:
    
    - 2020: $22.67 trillion
    - 2021: $23.32 trillion
    - 2022: $24.05 trillion


<a id='4'></a>

<a id='4'></a>
## 4 - Experimenting with your RAG System
---

Now you can experiment with your own queries to see the system in action! You can write any query, and it will display answers both with and without RAG. Keep in mind that the dataset you're working with is related to news data from 2024, so not all queries will be effective in demonstrating the framework. Some example queries you might try include:

* What were the most important events of the past year?
* How is global warming progressing in 2024?
* Tell me about the most recent advances in AI.
* Give me the most important facts from past year.

You can also specify a layout for the augmented prompt that includes placeholders for {query} and {documents} to indicate where they should be inserted within your prompt structure. For example:

```
This is the query: {query}
These are the documents: {documents}
```


```python
display_widget(llm_call)
```


    HTML(value='\n    <style>\n        .custom-output {\n            background-color: #f9f9f9;\n            color…



    Text(value='', description='Query:', layout=Layout(width='100%'), placeholder='Type your query here')



    Textarea(value='', description='Augmented prompt layout:', layout=Layout(height='100px', width='100%'), placeh…



    IntSlider(value=5, description='Top K:', max=20, min=1, style=SliderStyle(description_width='initial'))



    Button(description='Get Responses', style=ButtonStyle(button_color='#f0f0f0'))



    Output()



    HBox(children=(Label(value='With RAG', layout=Layout(width='45%')), Label(value='Without RAG', layout=Layout(w…



    HBox(children=(Output(layout=Layout(border_bottom='1px solid #ccc', border_left='1px solid #ccc', border_right…



