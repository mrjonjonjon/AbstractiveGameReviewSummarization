from wtpsplit import WtP
import util
#wtp = WtP("wtp-canine-s-12l")
wtp = WtP("wtp-bert-mini")

# optionally run on GPU for better performance
# also supports TPUs via e.g. wtp.to("xla:0"), in that case pass `pad_last_batch=True` to wtp.split
#wtp.to("cuda")

# returns ["This is a test", "This is another test."]
x=wtp.split('i like dogs i also like cats they make me so happy',lang_code='en')
print(x)