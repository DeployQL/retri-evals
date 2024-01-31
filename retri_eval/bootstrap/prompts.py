import dspy


class RetrievalTask(dspy.Signature):
    """Given a document, generate a list of potential retrieval tasks that a user might be trying to solve when searching for this document. be creative"""

    document = dspy.InputField(prefix="Document:")
    tasks = dspy.OutputField(
        # prefix="Tasks:",
        desc="A comma separated list",
        format=lambda x: ", ".join(x) if isinstance(x, list) else x,
    )


class IntentSpecificQuery(dspy.Signature):
    """Given a document, generate queries for each query type. The query should help answer the retrieval task provided."""

    document = dspy.InputField()
    query_types = dspy.InputField(
        prefix="Query Type:",
        desc="The format of the query. For instance, a query could be short or long, or just contain keywords",
    )
    retrieval_task = dspy.InputField(
        prefix="Retrieval Task:",
        desc="The retrieval task that the query should help answer.",
    )
    query = dspy.OutputField()


class RelevantQuery(dspy.Signature):
    """Retrieve a Query answered by the following Document."""

    document = dspy.InputField()
    query = dspy.OutputField(desc="often a question that's more than 3 words.")


class NotRelevantQuery(dspy.Signature):
    """Given a document, write a bad question that seems related to the document, but the document can't answer."""

    document = dspy.InputField()
    query = dspy.OutputField(desc="often a question that's more than 3 words.")


class RankedDocumentList(dspy.Signature):
    """Given a query, return a ranked list of documents. The first document should be the most relevant to the query, and each following
    document is less relevant than the previous."""

    query = dspy.InputField(
        prefix="Query:",
    )
    gold_document = dspy.InputField(
        prefix="Gold answer: ",
        desc="The document that the query is supposed to retrieve",
    )
    documents = dspy.OutputField(
        prefix="Ranked Documents:",
        desc="A list of documents, ranked by relevance to the query. Comma separated.",
        format=lambda x: ", ".join(x) if isinstance(x, list) else x,
    )


class QueryDocumentPair(dspy.Signature):
    """Given a query, generate a document that is relevant to the query and a hard negative document that is not relevant."""

    query = dspy.InputField(
        prefix="Query:",
    )
    relevant_document = dspy.OutputField(prefix="Relevant Document:")
    not_relevant_document = dspy.OutputField(prefix="Not Relevant Document:")


# class InferRetrievalTask(dspy.Signature):
#     """Given a document, brainstorm possible tasks that a user might be trying to solve with this document."""


class CoT(dspy.Module):
    def __init__(self):
        super().__init__()
        self.generate_answer = dspy.ChainOfThought(RelevantQuery)

    def forward(self, document: str):
        return self.generate_answer(document=document)


# this follows UDAPDR prompting.
def generate_udapdr_query(given_passage, prompt_number, given_good_question, llm):
    given_passage = " ".join(given_passage.split(" ")[:192])

    if prompt_number == 0:
        given_prompt = "Write a Question answered by the given Passage.\n"  # passage
        given_prompt += (
            "Passage: " + given_passage + "\n"
        )  # " ".join(given_passage.split(" ")[:256])
        given_prompt += "Question:"  # passage
    elif prompt_number == 1:
        given_prompt = "Example 1:\n"
        given_prompt += "Document: We don't know a lot about the effects of caffeine during pregnancy on you and your baby. So it's best to limit the amount you get each day. If you are pregnant, limit caffeine to 200 milligrams each day. This is about the amount in 1½ 8-ounce cups of coffee or one 12-ounce cup of coffee.\n"
        given_prompt += (
            "Good Question: How much caffeine is ok for a pregnant woman to have?\n"
        )
        given_prompt += "Bad Question: Is a little caffeine ok during pregnancy?\n\n"
        given_prompt += "Example 2:\n"
        given_prompt += "Document: Passiflora herbertiana. A rare passion fruit native to Australia. Fruits are green-skinned, white fleshed, with an unknown edible rating. Some sources list the fruit as edible, sweet and tasty, while others list the fruits as being bitter and inedible.\n"
        given_prompt += "Good Question: What is Passiflora herbertiana (a rare passion fruit) and how does it taste like?\n"
        given_prompt += "Bad Question: What fruit is native to Australia?\n\n"
        given_prompt += "Example 3:\n"
        given_prompt += "Document: The Canadian Armed Forces. 1 The first large-scale Canadian peacekeeping mission started in Egypt on November 24, 1956. 2 There are approximately 65,000 Regular Force and 25,000 reservist members in the Canadian military. 3 In Canada, August 9 is designated as National Peacekeepers' Day.\n"
        given_prompt += (
            "Good Question: Information on the Canadian Armed Forces size and history\n"
        )
        given_prompt += "Bad Question: How large is the Canadian military?\n\n"
        given_prompt += "Example 4:\n"
        given_prompt += "Document: " + given_passage + "\n"  # + "\n"
        given_prompt += "Good Question:"
    elif prompt_number == 2:
        given_prompt = "Example 1:\n"
        given_prompt += "Document: We don't know a lot about the effects of caffeine during pregnancy on you and your baby. So it's best to limit the amount you get each day. If you are pregnant, limit caffeine to 200 milligrams each day. This is about the amount in 1½ 8-ounce cups of coffee or one 12-ounce cup of coffee\n"
        given_prompt += "Relevant Query: Is a little caffeine ok during pregnancy?\n"
        given_prompt += "Example 2:\n"
        given_prompt += "Document: Passiflora herbertiana. A rare passion fruit native to Australia. Fruits are green-skinned, white fleshed, with an unknown edible rating. Some sources list the fruit as edible, sweet and tasty, while others list the fruits as being bitter and inedible.\n"
        given_prompt += "Relevant Query: What fruit is native to Australia?\n"
        given_prompt += "Example 3:\n"
        given_prompt += "Document: The Canadian Armed Forces. 1 The first large-scale Canadian peacekeeping mission started in Egypt on November 24, 1956. 2 There are approximately 65,000 Regular Force and 25,000 reservist members in the Canadian military. 3 In Canada, August 9 is designated as National Peacekeepers' Day\n"
        given_prompt += "Relevant Query: How large is the Canadian military?\n"
        given_prompt += "Example 4:\n"
        given_prompt += "Document: " + given_passage + "\n"
        given_prompt += "Relevant Query:"
    elif prompt_number == 3:
        given_prompt = (
            "Retrieve a Query answered by the following Document.\n"  # passage
        )
        given_prompt += "Document: " + given_passage + "\n"
        given_prompt += "Query:"
    elif prompt_number == 4:
        given_prompt = (
            "Design a Question that is answered by the following Passage.\n"  # passage
        )
        given_prompt += "Passage: " + given_passage + "\n"
        given_prompt += "Question:"
    elif prompt_number == -1:
        given_prompt = "Example 1:\n"
        given_prompt += "Document: We don't know a lot about the effects of caffeine during pregnancy on you and your baby. So it's best to limit the amount you get each day. If you are pregnant, limit caffeine to 200 milligrams each day. This is about the amount in 1½ 8-ounce cups of coffee or one 12-ounce cup of coffee.\n"
        given_prompt += (
            "Good Question: How much caffeine is ok for a pregnant woman to have?\n"
        )
        given_prompt += "Bad Question: Is a little caffeine ok during pregnancy?\n\n"
        given_prompt += "Example 2:\n"
        given_prompt += "Document: Passiflora herbertiana. A rare passion fruit native to Australia. Fruits are green-skinned, white fleshed, with an unknown edible rating. Some sources list the fruit as edible, sweet and tasty, while others list the fruits as being bitter and inedible.\n"
        given_prompt += "Good Question: What is Passiflora herbertiana (a rare passion fruit) and how does it taste like?\n"
        given_prompt += "Bad Question: What fruit is native to Australia?\n\n"
        given_prompt += "Example 3:\n"
        given_prompt += "Document: The Canadian Armed Forces. 1 The first large-scale Canadian peacekeeping mission started in Egypt on November 24, 1956. 2 There are approximately 65,000 Regular Force and 25,000 reservist members in the Canadian military. 3 In Canada, August 9 is designated as National Peacekeepers' Day.\n"
        given_prompt += (
            "Good Question: Information on the Canadian Armed Forces size and history\n"
        )
        given_prompt += "Bad Question: How large is the Canadian military?\n\n"
        given_prompt += "Example 4:\n"
        given_prompt += "Document: " + given_passage + "\n"  # + "\n"
        given_prompt += "Good Question: " + given_good_question + "\n"
        given_prompt += "Bad Question:"

    ########################################################

    response = llm.basic_request(prompt=given_prompt, n_predict=32)[0]
    query = response.replace("\n", "").replace("\t", "")

    ########################################################

    if prompt_number == 1:
        if query.lower().find("bad question") != -1:
            bad_question_index = query.find("bad question")
            query = query[:bad_question_index]
        query = query.replace("good question", "").replace(": ", "")

    elif prompt_number == -1:
        if query.lower().find("bad question") != -1:
            bad_question_index = query.find("bad question")
            query = query[bad_question_index:]
        query = query.replace("bad question", "").replace(": ", "")

    if query.find("?") != -1:
        question_mark_index = query.find("?")
        query = query[: question_mark_index + 1]
    query = query.strip()

    #########################################################

    return query
