from BLEU import *


def run():
    source = '正因为你为你的玫瑰花费了时间,这才使你的玫瑰变得如此重要'  # source
    target = 'What makes your rose so important is the time you have wasted for it.'  # target
    inference = 'It is the time you have wasted for your rose that makes your rose so important.'  # inference
    bleu1 = BLEU(source, target, inference)
    bleu1.evaluate()

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    run()
