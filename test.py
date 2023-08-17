from langchain.llms import OpenLLM

def main():
    llm = OpenLLM(model_name="falcon", model_id='tiiuae/falcon-7b-instruct')

    llm("What is the difference between a duck and a goose? And why there are so many Goose in Canada?")


if __name__ == "__main__":
    main()