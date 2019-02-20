from qa_engine.base import QABase
from qa_engine.score_answers import main as score_answers
import nltk

base = __import__('baseline-stub')

stopwords = set(nltk.corpus.stopwords.words("english"))
GRAMMAR = """
            N: {<PRP>|<NN.*>}
            V: {<V.*>}
            ADJ: {<JJ.*>}
            NP: {<DT>? <ADJ>* <N>+}
            PP: {<IN> <NP>}
            VP: {<TO>? <V> (<NP>|<PP>)*}
            """

chunker = nltk.RegexpParser(GRAMMAR)
lmtzr = nltk.stem.WordNetLemmatizer()
LOC_PP = {"in", "on", "at", "to"}
WHY_PP = {"because", "as"}
WHAT_PP = {"the", "a"}


def loc_filter(subtree):
    return subtree.label() == "PP"


def why_filter(subtree):
    return subtree.label() == "N"


def what_filter(subtree):
    return subtree.label() == "N" or subtree == "NP"


def get_answer_phrase(question, sentence):
    """
    Given a question and the sentence with an answer, extract the answer.
    :param question: Question asked of us.
    :param sentence: Sentence with question in it.
    :return: string phrase of answer
    """

    # Tokenize question for W-word
    q_toks = nltk.word_tokenize(question)
    sent_toks = nltk.word_tokenize(sentence)
    sent_pos = nltk.pos_tag(sent_toks)
    tree = chunker.parse(sent_pos)

    q_toks = [word.lower() for word in q_toks]

    if "where" in q_toks:
        # print("q_toks contains where!", q_toks)
        set_to_use = LOC_PP
        filter_to_use = loc_filter
    elif "why" in q_toks:
        # print("q_toks contains why!", q_toks)
        set_to_use = WHY_PP
        filter_to_use = why_filter
    else:
        # print("else!", q_toks)
        set_to_use = WHAT_PP
        filter_to_use = what_filter

    answer_list = []
    for subtree in tree.subtrees(filter=filter_to_use):
        # print(subtree)
        if subtree[0][0].lower() in set_to_use:
            # print("appending", subtree[0][0][0], " as it is in set_to_use")
            answer_list.append(subtree)

    final_answer = " ".join([token[0] for token in answer_list[0].leaves()]) \
        if len(answer_list) > 0 else sentence
    return final_answer


def get_answer(question, story):
    """
    :param question: dict
    :param story: dict
    :return: str


    question is a dictionary with keys:
        dep -- A list of dependency graphs for the question sentence.
        par -- A list of constituency parses for the question sentence.
        text -- The raw text of story.
        sid --  The story id.
        difficulty -- easy, medium, or hard
        type -- whether you need to use the 'sch' or 'story' versions
                of the .
        qid  --  The id of the question.


    story is a dictionary with keys:
        story_dep -- list of dependency graphs for each sentence of
                    the story version.
        sch_dep -- list of dependency graphs for each sentence of
                    the sch version.
        sch_par -- list of constituency parses for each sentence of
                    the sch version.
        story_par -- list of constituency parses for each sentence of
                    the story version.
        sch --  the raw text for the sch version.
        text -- the raw text for the story version.
        sid --  the story id


    """
    ###     Your Code Goes Here         ###
    version = question["type"]

    question_text = question["text"]

    # prefer scheherazade
    if "Sch" in version:
        text = story['sch']
    else:
        text = story['text']

    question = base.get_sentences(question_text)

    text = base.get_sentences(text)

    answer = base.baseline(question[0], text, stopwords)

    answer_text = ""
    for (x, y) in answer:
        answer_text += (" " if x[0].isalnum() else "") + x
    print("Question:", question_text + "\n")
    print("Answer:", answer_text + "\n\n")

    ###     End of Your Code         ###
    answer = get_answer_phrase(question_text, answer_text)
    return answer


#############################################################
###     Dont change the code in this section
#############################################################
class QAEngine(QABase):
    @staticmethod
    def answer_question(question, story):
        answer = get_answer(question, story)
        return answer


def run_qa():
    QA = QAEngine()
    QA.run()
    QA.save_answers()


#############################################################


def main():
    run_qa()
    # You can uncomment this next line to evaluate your
    # answers, or you can run score_answers.py
    score_answers()


if __name__ == "__main__":
    main()
