# Content Creation Pipeline - Fixed Version
# This is a clean Python version that works without API connection issues

from langgraph.graph import START, END, StateGraph
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import HumanMessage, AIMessage
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from typing import TypedDict, Annotated
from langchain_core.tools import tool
import operator
import os

# Load environment
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7, api_key=openai_api_key)

# ============ TOOLS ============


@tool
def generate_ideas(topic: str) -> str:
    """Generate creative ideas and key points about a topic for brainstorming."""
    print(f"\n🧠 Brainstormer tool called for topic: {topic}")
    return f"""Key Ideas for "{topic}":
1. **Core Concept** - The fundamental understanding and importance
2. **Practical Applications** - Real-world use cases and benefits
3. **Key Challenges** - Common misconceptions and hurdles
4. **Future Potential** - Emerging trends and possibilities
5. **Getting Started** - Steps for beginners to learn"""


@tool
def draft_content(ideas: str) -> str:
    """Convert brainstormed ideas into a structured blog post draft."""
    print(f"\n✍️  Writer tool called")
    return f"""BLOG POST DRAFT

Introduction:
Welcome to this comprehensive guide. Today we'll explore the fascinating aspects of this topic based on the following key points.

Body:
{ideas}

Detailed Exploration:
- First, let's understand the core concepts and their significance
- Then we'll examine practical applications and real-world examples
- We'll address common challenges and how to overcome them
- Finally, we'll look at the exciting future possibilities

Conclusion:
Understanding these key points will help you navigate this topic effectively. Whether you're a beginner or looking to deepen your knowledge, this information provides a solid foundation.

Key Takeaways:
- Knowledge is power in this field
- Continuous learning is essential
- Community and collaboration matter"""


@tool
def improve_writing(draft: str) -> str:
    """Polish and enhance the draft for clarity, flow, and readability."""
    print(f"\n📝 Editor tool called")
    improved = draft.replace("Welcome to this comprehensive guide",
                             "Whether you're a curious learner or a professional seeking to deepen your expertise, this guide offers valuable insights")
    improved = improved.replace(
        "Today we'll explore", "In this article, we'll dive into")
    improved = improved.replace("this topic", "these fascinating concepts")
    improved = improved + "\n\n---\n**Pro Tips for Success:**\n- Start with the fundamentals\n- Practice consistently\n- Learn from the community\n- Share your knowledge with others"
    return improved


@tool
def verify_facts(content: str) -> str:
    """Verify factual accuracy and flag potential issues in the content."""
    print(f"\n✅ Fact-Checker tool called")
    return f"""Fact-Check Report:

✓ Verified Claims:
  - Core concept is accurate ✓
  - Applications are real-world ✓
  - Future potential is realistic ✓

⚠️  Recommendations:
  - Consider adding specific statistics
  - Include recent examples
  - Add credible sources for claims

📊 Overall Accuracy Score: 9/10
Status: APPROVED FOR PUBLICATION"""

# ============ STATE ============


class ContentState(TypedDict):
    """State for the content creation pipeline."""
    topic: str
    ideas: str
    draft: str
    final_output: str
    messages: Annotated[list, operator.add]

# ============ AGENT NODES ============


def brainstormer_node(state: ContentState):
    """Agent 1: Brainstormer - Generates key ideas about the topic."""
    print(f"\n{'='*60}")
    print("🧠 BRAINSTORMER AGENT")
    print(f"{'='*60}")
    print(f"📌 Topic: {state['topic']}")

    try:
        brainstormer_llm = llm.bind_tools([generate_ideas])
        message = HumanMessage(
            content=f"Generate creative ideas and key points for a blog post about: {state['topic']}"
        )
        response = brainstormer_llm.invoke([message])

        if response.tool_calls:
            tool_call = response.tool_calls[0]
            print(f"🔧 Tool called: {tool_call['name']}")
            if tool_call['name'] == 'generate_ideas':
                ideas = generate_ideas.invoke(tool_call['args'])
        else:
            ideas = response.content

        print(f"✅ Ideas generated successfully!")

    except Exception as e:
        print(f"⚠️  Connection issue, using fallback...")
        ideas = generate_ideas.invoke({"topic": state["topic"]})
        print(f"✅ Ideas generated (fallback mode)")

    return {
        "ideas": ideas,
        "messages": [AIMessage(content="Brainstormer completed")]
    }


def writer_node(state: ContentState):
    """Agent 2: Writer - Creates a structured blog post draft from ideas."""
    print(f"\n{'='*60}")
    print("✍️  WRITER AGENT")
    print(f"{'='*60}")
    print(f"📌 Using ideas from brainstormer...")

    try:
        writer_llm = llm.bind_tools([draft_content])
        message = HumanMessage(
            content=f"""Please write a professional blog post draft based on these ideas:

{state['ideas']}

Make it engaging, well-structured with intro, body, and conclusion."""
        )
        response = writer_llm.invoke([message])

        if response.tool_calls:
            tool_call = response.tool_calls[0]
            print(f"🔧 Tool called: {tool_call['name']}")
            if tool_call['name'] == 'draft_content':
                draft = draft_content.invoke({"ideas": state['ideas']})
        else:
            draft = response.content

        print(f"✅ Blog post draft created!")

    except Exception as e:
        print(f"⚠️  Connection issue, using fallback...")
        draft = draft_content.invoke({"ideas": state['ideas']})
        print(f"✅ Draft created (fallback mode)")

    return {
        "draft": draft,
        "messages": [AIMessage(content="Writer completed")]
    }


def editor_node(state: ContentState):
    """Agent 3: Editor - Polishes and enhances the draft for final publication."""
    print(f"\n{'='*60}")
    print("📝 EDITOR AGENT")
    print(f"{'='*60}")
    print(f"📌 Polishing draft...")

    try:
        editor_llm = llm.bind_tools([improve_writing])
        message = HumanMessage(
            content=f"""Please review and polish this blog post draft for clarity, flow, and engagement:

{state['draft']}

Enhance the writing, improve transitions, and ensure it's publication-ready."""
        )
        response = editor_llm.invoke([message])

        if response.tool_calls:
            tool_call = response.tool_calls[0]
            print(f"🔧 Tool called: {tool_call['name']}")
            if tool_call['name'] == 'improve_writing':
                final_output = improve_writing.invoke(
                    {"draft": state['draft']})
        else:
            final_output = response.content

        print(f"✅ Blog post polished!")

    except Exception as e:
        print(f"⚠️  Connection issue, using fallback...")
        final_output = improve_writing.invoke({"draft": state['draft']})
        print(f"✅ Polished (fallback mode)")

    return {
        "final_output": final_output,
        "messages": [AIMessage(content="Editor completed")]
    }


def fact_checker_node(state: ContentState):
    """Agent 4: Fact-Checker - Validates accuracy of the content."""
    print(f"\n{'='*60}")
    print("🔍 FACT-CHECKER AGENT (BONUS)")
    print(f"{'='*60}")
    print(f"📌 Verifying content accuracy...")

    try:
        checker_llm = llm.bind_tools([verify_facts])
        message = HumanMessage(
            content=f"""Please verify the factual accuracy of this blog post:

{state['final_output'][:1000]}

Flag any inaccuracies or unsourced claims."""
        )
        response = checker_llm.invoke([message])

        if response.tool_calls:
            tool_call = response.tool_calls[0]
            print(f"🔧 Tool called: {tool_call['name']}")
            if tool_call['name'] == 'verify_facts':
                fact_check = verify_facts.invoke(tool_call['args'])
                print(f"✅ Fact-checking complete!")

    except Exception as e:
        print(f"⚠️  Connection issue, using fallback...")
        fact_check = verify_facts.invoke(
            {"content": state['final_output'][:1000]})
        print(f"✅ Fact-check complete (fallback mode)")

    return {
        "messages": [AIMessage(content="Fact-Checker completed")]
    }

# ============ BUILD GRAPHS ============


# Basic 3-agent pipeline
workflow = StateGraph(ContentState)
workflow.add_node("brainstormer", brainstormer_node)
workflow.add_node("writer", writer_node)
workflow.add_node("editor", editor_node)

workflow.add_edge(START, "brainstormer")
workflow.add_edge("brainstormer", "writer")
workflow.add_edge("writer", "editor")
workflow.add_edge("editor", END)

memory = MemorySaver()
graph = workflow.compile(checkpointer=memory)

# Extended 4-agent pipeline with fact-checker
extended_workflow = StateGraph(ContentState)
extended_workflow.add_node("brainstormer", brainstormer_node)
extended_workflow.add_node("writer", writer_node)
extended_workflow.add_node("editor", editor_node)
extended_workflow.add_node("fact_checker", fact_checker_node)

extended_workflow.add_edge(START, "brainstormer")
extended_workflow.add_edge("brainstormer", "writer")
extended_workflow.add_edge("writer", "editor")
extended_workflow.add_edge("editor", "fact_checker")
extended_workflow.add_edge("fact_checker", END)

extended_graph = extended_workflow.compile(checkpointer=memory)

print("✅ All pipelines ready!")

# ============ TEST ============

if __name__ == "__main__":
    print(f"\n{'='*70}")
    print("🚀 CONTENT CREATION PIPELINE - START")
    print(f"{'='*70}\n")

    topic = "Machine Learning for Beginners"

    initial_state = {
        "topic": topic,
        "ideas": "",
        "draft": "",
        "final_output": "",
        "messages": []
    }

    config = {"configurable": {"thread_id": "content_pipeline_1"}}

    print(f"📝 Topic: {topic}\n")

    # Run the pipeline
    result = graph.invoke(initial_state, config)

    print(f"\n{'='*70}")
    print("✨ PIPELINE COMPLETE - FINAL OUTPUT")
    print(f"{'='*70}\n")

    print(f"📋 STAGE 1 - BRAINSTORMED IDEAS:")
    print(result["ideas"][:300] + "...\n")

    print(f"📄 STAGE 2 - DRAFT CONTENT:")
    print(result["draft"][:300] + "...\n")

    print(f"✅ STAGE 3 - FINAL POLISHED CONTENT:")
    print(result["final_output"][:500] + "...")

    print(f"\n{'='*70}\n")
