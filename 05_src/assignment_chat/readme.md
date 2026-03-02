# Solution:

---

The chat client is built with a focus on providing assistance in financial analysis, resources for retired seniors and retirement planning
It has been built with three services as outlined in the assignment

1. API - It uses the marketstack eod api to get the financial data for the particular stock. It then uses the python pandas-ta library to perform technical analysis and present the results to the user as summary
2. RAG - It uses the resources prepared by Ontario government for Seniors which details different programs for Seniors. It reads the pdf file, chunks the data, encrypts the data and stores it in a persistent Chromadb database.
3. Web Search - It uses tavily to search the web for any resources which would not be covered by the above mentioned tools

The chat application has been built with a system prompt, to not provide response on any restricted topics like Taylor Swift, Horoscope, etc which was part of the assignment
It also uses a strict rules in the prompt which ensures that there is no jailbreaking and it provides a disclaimer that the financial information is provided as a guidance and not a legal advise.

One of my interesting find was, if the chat client is asked to look for data from the past 3 months, it would pick dates from 2023, which I believe was the last knowledge of the llm when it was trained. To overcome this, I injected current_date variable which provides the model with the present date to provide more accurate results.

Additional library added:"ft-pandas-ta==0.3.16"

# Resources:

---

https://thelocal.to/aging-seniors-benefits-tax-credits-programs-rebates-toronto/
https://www.canada.ca/en/employment-social-development/campaigns/seniors.html
https://www.canada.ca/en/services/life-events/retirement/prepare.html

# Questions:

## Below are set of questions which have been tested with the chat client

Can you give me the technical analysis for msft stock based on the last 2 months of data
Can you give me the technical analysis for Microsoft stock based on the last 2 months of data
Can you help me analyse my investment style based on my risk assesment
What are the different programs from the government for seniors in Ontario
Can you help me in planning my retirement. Provide me details on what should be my financial goals if I plan to retire when I am 65 years. My present annual salary is 100K and I annually save 20K. Please provide me a detailed steps on investment strategies and plans
What are some investment trends for 2026 if I would want to invest longterm for retirement
