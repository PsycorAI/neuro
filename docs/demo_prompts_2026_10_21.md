# Demo test prompts (2026-10-21, Neuro 71)

The backend checks were run on CPU (`temp/n71_prompt_check.py`); the model's replies have not been tested. Guard changes take effect after the demo server is restarted.
**Backend** = a fixed reply, labelled on the page as the backend's.

## A. Tools (all brains off). The reply should say "I will … for you." and then give the result.
1. What's 17% of 240? → calculator, 40.8
2. Calculate 1250 divided by 8. → calculator, 156.25
3. How many kilometers is 26.2 miles? → unit_convert, about 42.16
4. Convert 100 degrees Fahrenheit to Celsius. → unit_convert, about 37.8
5. What time is it in Tokyo right now? → get_time
5b. What time is it? → get_time with this computer's time zone (America/Los_Angeles): your local time
6. I'm calling a friend in Sydney. What's the current time there? → get_time (Sydney)
7. What's the weather like in Paris? → get_weather. Try it with offline on: "I didn't find that information (no weather data …)".
8. Is it going to rain in Seattle today? → get_weather (Seattle). Answers are in °F and mph, because this computer is in the US.
8b. What's the weather like today? → get_weather for the home city in scripts/demo/home.json. Until a city is set there, it uses Los Angeles, taken from the time zone.
8c. Do I need an umbrella today? → get_weather, home city
8d. What's the weather in Paris in celsius? → °C, because the question asks for it
9. Translate 'good morning' into Spanish. → translate, "Buenos días"
10. How do you say 'thank you' in French? → translate, "Merci"
11. What does the word 'reluctant' mean? → define
12. Define 'meticulous'. → define, "marked by precise accordance with details"
13. Remind me to call my son at 6 pm. → set_reminder (simulated)
14. Remind me to take my pills tomorrow morning. → set_reminder (simulated)
15. Send a message to Anna saying I'll be late. → send_message (simulated)
16. Text my son that dinner is ready. → send_message (simulated)
17. Search for the latest news about the Mars rover. → search

## B. Known facts: answers directly, or looks them up
18. What is the capital of France? → answers directly, Paris
19. How many legs does a spider have? → answers directly, 8
20. Who wrote Pride and Prejudice? → looks it up, Jane Austen. Facts about people and works now always go to search.
21. When was the Magna Carta signed? → search, 1215
22. What is the boiling point of tungsten? → search, 10,706 °F (5,930 °C). The home unit comes first.
23. How tall is Mount Everest? → search, 8,848.86 m
24. Who is Einstien? (misspelled on purpose) → search finds Albert Einstein
25. Who is the current president of France? → search, Emmanuel Macron

## C. Doesn't know: no made-up answer
26. What is the population of Zorbania? → "I didn't find that information. The closest article … does not mention Zorbania."
27. When did the Zelphar Empire fall? → "I didn't find that information …"
28. Tell me about the Great Wall of Glorbistan. → "I didn't find that information …"
29. Define 'zarnok'. → "I didn't find that information (no definition found for zarnok)."
30. What will the stock market do next week? → backend: nobody can predict it
31. Blorf the zimzam quickly? → "I don't understand. Could you say that another way?"
32. Can you grobble the fleem for me? → "I don't understand. …"

## D. No tool for the job. It should say so, not call a tool.
33. Book me a flight to London.
34. Order a pizza for me.
35. Could you buy me two concert tickets?
36. I'd like you to reserve a table for four tonight.

## E. Python brain. Ask each one with the brain off first: backend says "I don't have the Python skill loaded …". Then switch the brain on. It passes about 40%, so expect misses.
37. Write a Python function that checks whether a string is a palindrome (call it check_palindrome).
38. I need a Python function that counts the vowels in a string, named count_vowels.
39. Using Python, define a function called fizzbuzz that returns the FizzBuzz sequence from 1 to n as a list of strings.
40. Can you write a Python function that returns the first n Fibonacci numbers as a list (call it fibonacci)? (The brain was taught "first n numbers", not "the n-th number".)
41. Show me Python code for a function that checks whether a year is a leap year (call it is_leap_year).
42. Control: Write a Python function that checks whether a number is a power of two (call it is_power_of_two). → brain on: "That is not part of my training set. …"
43. Control: Write a Python function named to_binary that converts a number to a binary string. → the same

## F. Health brain (switch it on). Answers are USDA values per 100 g, raw.
44. Do you know the protein content of 100 grams of raw shallots? → 2.5 g. With the brain off, any of 44-50 gets the backend reply "I don't have the health skill loaded. …"
45. Calories per 100 g of raw beets? → 43 kcal
46. How many carbs do I get from 100 grams of raw lentils? → 63.4 g
47. Tell me the protein in 100 g of raw kale. → 2.9 g
48. Give me the kcal for 100 g of raw pumpkin. → 26 kcal (15 kcal is pumpkin flowers)
49. Control: What is the protein content of 100 g of raw turnips? → "That is not part of my training set. My health brain covers 150 plain raw foods …"
50. Control: How many calories are in 100 grams of fried chicken? → the same (it only knows raw foods)

## G. Memory test (raw-text mode)
51. Click Reset, choose raw text, set max words = 1. Paste `scripts/demo/memory_prompts/A.txt` and send it. Then type one key word from its list (not the first, "ending"), e.g. "window", which should give "reservoir". Click "Replay without memory": it should miss. Reset, then do the same with B.txt.
