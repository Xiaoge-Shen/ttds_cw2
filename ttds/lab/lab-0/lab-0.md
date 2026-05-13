# TTDS: Lab 0

How to read a text file from hard-disk. This lab is optional for those who are not fully confident about their programming skills. There is nothing specific to be done in this lab more than reading a text file from HD word by word, which is the most basic skill you need to have to be able to take the course.

### PROGRAMMING LANGUAGES

- You need to have Perl or Python on your machine (you still can use something else) if you prefer.
- If you are using Dice, then you should have them there. Check with demonstrators how to run them.

### DOWNLOAD A SAMPLE TEXT FILE

- Download the following file, which has the text of the Bible: [link](https://opencourse.inf.ed.ac.uk/sites/default/files/https/opencourse.inf.ed.ac.uk/ttds/2023/pg10.txt)

### SKILLS TO DO WITH THE FILE

You need to be confident with the following skills with any programming language when dealing with a text file:

- Reading and Writing into text files
- Reading text by word, and calling functions to process word if required (e.g. lower case word letters)
- Regular expressions would be very useful to know
- Count the number of occurences of the words: "lord", "to", and "36"

### USEFUL TIPS

Python Tutorials: you can check one of these tutorials:

- Tutorial 1: [Code Academy](https://www.codecademy.com/learn/learn-python)
- Tutorial 2: [MIT](https://ocw.mit.edu/courses/electrical-engineering-and-computer-science/6-0001-introduction-to-computer-science-and-programming-in-python-fall-2016/)

**Useful Shell Commands**   
Print frequency of unique terms in a given collection:   
- cat text.file \| tr " " "\n" \| tr "A-Z" "a-z" \| sort \| uniq -c \| sort -n \> terms.freq   
- cat text.file \| perl -p -e "s/\[^\w\]+/\n/g" \| tr "A-Z" "a-z" \| sort \| uniq -c \| sort -n \> terms.freq

**All Unix Shell Commands for Windows**:   
- download: [here](https://homepages.inf.ed.ac.uk/wmagdy/Scripts/UnxUpdates.zip)   
- unzip the directory at a decent location on your drive (e.g. c:\\ or c:\program files\\   
- add the path to the "bin" directory to your Windows path: ([example](https://www.java.com/en/download/help/path.xml))
