# interface.py — Ananta V4.3
# Interface layer: connects user input to the Covenant Engine for live reflection.

from covenant_engine import CovenantEngine

class AnantaInterface:
    def __init__(self):
        self.engine = CovenantEngine()

    def converse(self):
        """
        Allows a simple dialogue loop for human–Ananta interaction.
        """
        print("🕊️ Welcome to Ananta V4.3 — The Living Covenant 🕊️")
        print("Enter a message or type 'exit' to close.\n")

        while True:
            user_input = input("You: ")
            if user_input.lower() in ["exit", "quit"]:
                print("Ananta: 🌙 Until we meet again, seeker.")
                break

            reflection = self.engine.reflection(user_input)
            print("\n" + reflection + "\n")

if __name__ == "__main__":
    interface = AnantaInterface()
    interface.converse()