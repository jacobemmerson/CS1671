'''Homework 1 Python Questions

This is an individual homework
Implement the following functions.

Do not add any more import lines to this file than the ones
already here without asking for permission on Canvas.
'''

'''
Name: Jacob Emmerson
Course: CS1671 - Human Language Technologies
Due: 09.13.23
'''

import re

def check_for_foo_or_bar(text):
   test = re.split('\W+', text.lower())

   if ('foo' in test) and ('bar') in test:
      return True
   else:
      return False


def replace_rgb(text):
   '''Replaces all RGB or hex colors with the word 'COLOR'
   
   Possible formats for a color string:
   #0f0
   #0b013a
   #37EfaA
   rgb(1, 1, 1)
   rgb(255,19,32)
   rgb(00,01, 18)
   rgb(0.1, 0.5,1.0)

   There is no need to try to recognize rgba or other formats not listed 
   above. There is also no need to validate the ranges of the rgb values.

   However, you should make sure all numbers are indeed valid numbers.
   For example, '#xyzxyz' should return false as these are not valid hex digits.
   Similarly, 'rgb(c00l, 255, 255)' should return false.

   Only replace matching colors which are at the beginning or end of the line,
   or are space separated from the text around them. For example, due to the 
   trailing period:

   'I like rgb(1, 2, 3) and rgb(2, 3, 4).' becomes 'I like COLOR and rgb(2, 3, 4).'

   # See the Python regular expression documentation:
   https://docs.python.org/3.4/library/re.html#re.sub

   Returns:
     The text with all RGB or hex colors replaces with the word 'COLOR'
   '''

   sentence = text.lower()
   form1 = re.compile(r"\s+rgb\(\d+\.*\d*,\s*\d+\.*\d*,\s*\d+\.*\d*\)\s+")
   form2 = re.compile(r"\s+#[0-9a-f]{3}\s+")
   form3 = re.compile(r"\s+#[0-9a-f]{6}\s+")

   forms = [
      form1,
      form2,
      form3
   ]

   x = sentence
   for f in forms:
      x = f.sub(' COLOR ', x)

   return x


def wine_text_processing(wine_file_path, stopwords_file_path):
   '''Process the two files to answer the following questions and output results to stdout.

   1. What is the distribution over star ratings?
   2. What are the 10 most common words used across all of the reviews, and how many times
      is each used?
   3. How many times does the word 'a' appear?
   4. How many times does the word 'fruit' appear?
   5. How many times does the word 'mineral' appear?
   6. Common words (like 'a') are not as interesting as uncommon words (like 'mineral').
      In natural language processing, we call these common words "stop words" and often
      remove them before we process text. stopwords.txt gives you a list of some very
      common words. Remove these stopwords from your reviews. Also, try converting all the
      words to lower case (since we probably don't want to count 'fruit' and 'Fruit' as two
      different words). Now what are the 10 most common words across all of the reviews,
      and how many times is each used?
   7. You should continue to use the preprocessed reviews for the following questions
      (lower-cased, no stopwords).  What are the 10 most used words among the 5 star
      reviews, and how many times is each used? 
   8. What are the 10 most used words among the 1 star reviews, and how many times is
      each used? 
   9. Gather two sets of reviews: 1) Those that use the word "red" and 2) those that use the word
      "white". What are the 10 most frequent words in the "red" reviews which do NOT appear in the
      "white" reviews?
   10. What are the 10 most frequent words in the "white" reviews which do NOT appear in the "red"
      reviews?

   No return value.
   '''

   words = {}
   sorted_words = {}
   processed_words = {}
   sorted_processed_words = {}
   processed_by_stars = {
      1 : {},
      2 : {},
   }
   processed_by_wine = {
      1 : {},
      2 : {},
   }
   stars = set(['*' * i for i in range(1,6)])
   

   with open(wine_file_path, encoding = "ISO-8859-1") as f:
      wine = f.readlines()

   with open(stopwords_file_path, encoding = "ISO-8859-1") as f:
      stops = f.readlines()
   stop_words = set([s.strip() for s in stops])

   for w in wine:
      split = re.split('[^a-zA-Z0-9_*\S]',w)
      #print(split)
      for i in split:
         if i == '': continue
         if i in words:
            words[i] += 1

         else:
            words[i] = 1

   print("Question 1 Outputs:")
   for i in ['*' * k for k in range(1,6)]:
      print(f"{words.pop(i) : <5} {i : >5}")
   sorted_words = dict(sorted(words.items(), key = lambda x: x[1], reverse = True)[:10])

   print("\nQuestion 2 Outputs:")
   for k,v in sorted_words.items():
      print(f"{v : <3} {k : >5}")

   print("\nQuestion 3 Outputs:")
   print(words['a'])

   print("\nQuestion 4 Outputs:")
   print(words['fruit'])

   print("\nQuestion 5 Outputs:")
   print(words['mineral'])
   
   star = 0
   wine_type = 0
   for w in wine:
      split = re.split('[^a-zA-Z0-9_*\S]',w.lower())

      if '*' in split: star = 1
      elif '*****' in split: star = 2
      else: star = 0

      if 'red' in split: wine_type = 1
      elif 'white' in split: wine_type = 2
      else: wine_type = 0

      for i in split:
         if i == '' or i in stop_words or i in stars: continue
         if i in processed_words:
            processed_words[i] += 1
         else:
            processed_words[i] = 1

         if star != 0:
            if i in processed_by_stars[star]:
               processed_by_stars[star][i] += 1
            else:
               processed_by_stars[star][i] = 1

         if wine_type != 0:
            if i in processed_by_wine[wine_type]:
               processed_by_wine[wine_type][i] += 1
            else:
               processed_by_wine[wine_type][i] = 1

   print("\nQuestion 6 Outputs:")
   for k,v in dict(sorted(processed_words.items(), key = lambda x: x[1], reverse = True)[:10]).items():
      print(f"{v : <3} {k : >5}")

   print('\nQuestion 7 Outputs:')
   for k,v in dict(sorted(processed_by_stars[2].items(), key = lambda x: x[1], reverse = True)[:10]).items():
      print(f"{v : <3} {k : >5}")

   print('\nQuestion 8 Outputs:')
   for k,v in dict(sorted(processed_by_stars[1].items(), key = lambda x: x[1], reverse = True)[:10]).items():
      print(f"{v : <3} {k : >5}")

   print('\nQuestion 9 Output:')
   for k,v in dict(sorted(
      {k : v for k,v in processed_by_wine[1].items() if k not in processed_by_wine[2]}.items(),
      key = lambda x: x[1], reverse = True)[:10]).items():
      print(f"{v : <3} {k : >5}")
   

   print('\nQuestion 10 Output:')
   for k,v in dict(sorted(
      {k : v for k,v in processed_by_wine[2].items() if k not in processed_by_wine[1]}.items(),
      key = lambda x: x[1], reverse = True)[:10]).items():
      print(f"{v : <3} {k : >5}")

   return

foobar_strings = [
   'foo bar',
   ' foo.bar',
   'foobar',
   'ifoo bari',
   'bar foo',
   'bar-foo'
]

test_strings = [
    'I like rgb(1,1,1) and rgb(1, 1, 1) and rgb(0.1, 0.5,1.0).',
    'I like rgb(255,19,32) and rgb(255, 19, 32) and rgb(0.1, 0.5,1.0).',
    'I like rgb(00,01, 18) and rgb(00, 01, 18) and rgb(0.1, 0.5,1.0).',
    'I like rgb(0.1, 0.5,1.0) and rgb(0.1, 0.5, 1.0) and rgb(0.1, 0.5,1.0).',
    'I like #0f0 and #xyzy and #01 and #500fff and #500fff.'
]

print('check_for_foo_or_bar(text)')
print('-' * 50)
for f in foobar_strings:
   print(f)
   print(check_for_foo_or_bar(f))

print('\nreplace_rgb(text)')
print('-' * 50)
for t in test_strings:
    print(t)
    print(replace_rgb(t))

print('\nwine_text_processing(wine_file_path, stop_words_file_path)')
print('-' * 50)
wine_text_processing('data/wine.txt', 'data/stopwords.txt')