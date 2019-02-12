{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "metadata": {},
   "outputs": [],
   "source": [
    "import nltk"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "metadata": {},
   "outputs": [],
   "source": [
    "#nltk.download_shell()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "metadata": {},
   "outputs": [],
   "source": [
    "messages = [line.rstrip() for line in open('smsspamcollection/SMSSpamCollection')]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "5574\n"
     ]
    }
   ],
   "source": [
    "print(len(messages))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "'ham\\tOk lar... Joking wif u oni...'"
      ]
     },
     "execution_count": 6,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "messages[1]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "\"ham\\tPlease don't text me anymore. I have nothing else to say.\""
      ]
     },
     "execution_count": 7,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "messages[100]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "0 ham\tGo until jurong point, crazy.. Available only in bugis n great world la e buffet... Cine there got amore wat...\n",
      "\n",
      "\n",
      "1 ham\tOk lar... Joking wif u oni...\n",
      "\n",
      "\n",
      "2 spam\tFree entry in 2 a wkly comp to win FA Cup final tkts 21st May 2005. Text FA to 87121 to receive entry question(std txt rate)T&C's apply 08452810075over18's\n",
      "\n",
      "\n",
      "3 ham\tU dun say so early hor... U c already then say...\n",
      "\n",
      "\n",
      "4 ham\tNah I don't think he goes to usf, he lives around here though\n",
      "\n",
      "\n",
      "5 spam\tFreeMsg Hey there darling it's been 3 week's now and no word back! I'd like some fun you up for it still? Tb ok! XxX std chgs to send, Â£1.50 to rcv\n",
      "\n",
      "\n",
      "6 ham\tEven my brother is not like to speak with me. They treat me like aids patent.\n",
      "\n",
      "\n",
      "7 ham\tAs per your request 'Melle Melle (Oru Minnaminunginte Nurungu Vettam)' has been set as your callertune for all Callers. Press *9 to copy your friends Callertune\n",
      "\n",
      "\n",
      "8 spam\tWINNER!! As a valued network customer you have been selected to receivea Â£900 prize reward! To claim call 09061701461. Claim code KL341. Valid 12 hours only.\n",
      "\n",
      "\n",
      "9 spam\tHad your mobile 11 months or more? U R entitled to Update to the latest colour mobiles with camera for Free! Call The Mobile Update Co FREE on 08002986030\n",
      "\n",
      "\n"
     ]
    }
   ],
   "source": [
    "for mess_no, messages in enumerate(messages[:10]):\n",
    "    print(mess_no, messages)\n",
    "    print('\\n')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "metadata": {},
   "outputs": [],
   "source": [
    "import pandas as pd"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 13,
   "metadata": {},
   "outputs": [],
   "source": [
    "messages = pd.read_csv('smsspamcollection/SMSSpamCollection', sep='\\t'\n",
    "                      , names=['label','messages'])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 14,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>label</th>\n",
       "      <th>messages</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>ham</td>\n",
       "      <td>Go until jurong point, crazy.. Available only ...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>ham</td>\n",
       "      <td>Ok lar... Joking wif u oni...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>spam</td>\n",
       "      <td>Free entry in 2 a wkly comp to win FA Cup fina...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>ham</td>\n",
       "      <td>U dun say so early hor... U c already then say...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>ham</td>\n",
       "      <td>Nah I don't think he goes to usf, he lives aro...</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "  label                                           messages\n",
       "0   ham  Go until jurong point, crazy.. Available only ...\n",
       "1   ham                      Ok lar... Joking wif u oni...\n",
       "2  spam  Free entry in 2 a wkly comp to win FA Cup fina...\n",
       "3   ham  U dun say so early hor... U c already then say...\n",
       "4   ham  Nah I don't think he goes to usf, he lives aro..."
      ]
     },
     "execution_count": 14,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "messages.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 15,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "<class 'pandas.core.frame.DataFrame'>\n",
      "RangeIndex: 5572 entries, 0 to 5571\n",
      "Data columns (total 2 columns):\n",
      "label       5572 non-null object\n",
      "messages    5572 non-null object\n",
      "dtypes: object(2)\n",
      "memory usage: 43.6+ KB\n"
     ]
    }
   ],
   "source": [
    " messages.info()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 18,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>label</th>\n",
       "      <th>messages</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>count</th>\n",
       "      <td>5572</td>\n",
       "      <td>5572</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>unique</th>\n",
       "      <td>2</td>\n",
       "      <td>5169</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>top</th>\n",
       "      <td>ham</td>\n",
       "      <td>Sorry, I'll call later</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>freq</th>\n",
       "      <td>4825</td>\n",
       "      <td>30</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "       label                messages\n",
       "count   5572                    5572\n",
       "unique     2                    5169\n",
       "top      ham  Sorry, I'll call later\n",
       "freq    4825                      30"
      ]
     },
     "execution_count": 18,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "messages.describe()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 21,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>label</th>\n",
       "      <th>messages</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>ham</td>\n",
       "      <td>Go until jurong point, crazy.. Available only ...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>ham</td>\n",
       "      <td>Ok lar... Joking wif u oni...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>spam</td>\n",
       "      <td>Free entry in 2 a wkly comp to win FA Cup fina...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>ham</td>\n",
       "      <td>U dun say so early hor... U c already then say...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>ham</td>\n",
       "      <td>Nah I don't think he goes to usf, he lives aro...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>5</th>\n",
       "      <td>spam</td>\n",
       "      <td>FreeMsg Hey there darling it's been 3 week's n...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>6</th>\n",
       "      <td>ham</td>\n",
       "      <td>Even my brother is not like to speak with me. ...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>8</th>\n",
       "      <td>spam</td>\n",
       "      <td>WINNER!! As a valued network customer you have...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>9</th>\n",
       "      <td>spam</td>\n",
       "      <td>Had your mobile 11 months or more? U R entitle...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>11</th>\n",
       "      <td>spam</td>\n",
       "      <td>SIX chances to win CASH! From 100 to 20,000 po...</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "   label                                           messages\n",
       "0    ham  Go until jurong point, crazy.. Available only ...\n",
       "1    ham                      Ok lar... Joking wif u oni...\n",
       "2   spam  Free entry in 2 a wkly comp to win FA Cup fina...\n",
       "3    ham  U dun say so early hor... U c already then say...\n",
       "4    ham  Nah I don't think he goes to usf, he lives aro...\n",
       "5   spam  FreeMsg Hey there darling it's been 3 week's n...\n",
       "6    ham  Even my brother is not like to speak with me. ...\n",
       "8   spam  WINNER!! As a valued network customer you have...\n",
       "9   spam  Had your mobile 11 months or more? U R entitle...\n",
       "11  spam  SIX chances to win CASH! From 100 to 20,000 po..."
      ]
     },
     "execution_count": 21,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "messages.groupby('label').head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 17,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead tr th {\n",
       "        text-align: left;\n",
       "    }\n",
       "\n",
       "    .dataframe thead tr:last-of-type th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr>\n",
       "      <th></th>\n",
       "      <th colspan=\"4\" halign=\"left\">messages</th>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th></th>\n",
       "      <th>count</th>\n",
       "      <th>unique</th>\n",
       "      <th>top</th>\n",
       "      <th>freq</th>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>label</th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>ham</th>\n",
       "      <td>4825</td>\n",
       "      <td>4516</td>\n",
       "      <td>Sorry, I'll call later</td>\n",
       "      <td>30</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>spam</th>\n",
       "      <td>747</td>\n",
       "      <td>653</td>\n",
       "      <td>Please call our customer service representativ...</td>\n",
       "      <td>4</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "      messages                                                               \n",
       "         count unique                                                top freq\n",
       "label                                                                        \n",
       "ham       4825   4516                             Sorry, I'll call later   30\n",
       "spam       747    653  Please call our customer service representativ...    4"
      ]
     },
     "execution_count": 17,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "messages.groupby('label').describe()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 22,
   "metadata": {},
   "outputs": [],
   "source": [
    "messages['length'] = messages['messages'].apply(len)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 23,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>label</th>\n",
       "      <th>messages</th>\n",
       "      <th>length</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>ham</td>\n",
       "      <td>Go until jurong point, crazy.. Available only ...</td>\n",
       "      <td>111</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>ham</td>\n",
       "      <td>Ok lar... Joking wif u oni...</td>\n",
       "      <td>29</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>spam</td>\n",
       "      <td>Free entry in 2 a wkly comp to win FA Cup fina...</td>\n",
       "      <td>155</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>ham</td>\n",
       "      <td>U dun say so early hor... U c already then say...</td>\n",
       "      <td>49</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>ham</td>\n",
       "      <td>Nah I don't think he goes to usf, he lives aro...</td>\n",
       "      <td>61</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "  label                                           messages  length\n",
       "0   ham  Go until jurong point, crazy.. Available only ...     111\n",
       "1   ham                      Ok lar... Joking wif u oni...      29\n",
       "2  spam  Free entry in 2 a wkly comp to win FA Cup fina...     155\n",
       "3   ham  U dun say so early hor... U c already then say...      49\n",
       "4   ham  Nah I don't think he goes to usf, he lives aro...      61"
      ]
     },
     "execution_count": 23,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "messages.head()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Visualizations"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 24,
   "metadata": {},
   "outputs": [],
   "source": [
    "import matplotlib.pyplot as plt\n",
    "import seaborn as sns"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 25,
   "metadata": {},
   "outputs": [],
   "source": [
    "%matplotlib inline"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 28,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "<matplotlib.axes._subplots.AxesSubplot at 0x1155fe30>"
      ]
     },
     "execution_count": 28,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "messages['length'].plot.hist(bins=150)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 29,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "count    5572.000000\n",
       "mean       80.489950\n",
       "std        59.942907\n",
       "min         2.000000\n",
       "25%        36.000000\n",
       "50%        62.000000\n",
       "75%       122.000000\n",
       "max       910.000000\n",
       "Name: length, dtype: float64"
      ]
     },
     "execution_count": 29,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "messages['length'].describe()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 32,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>label</th>\n",
       "      <th>messages</th>\n",
       "      <th>length</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>1085</th>\n",
       "      <td>ham</td>\n",
       "      <td>For me the love should start with attraction.i...</td>\n",
       "      <td>910</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "     label                                           messages  length\n",
       "1085   ham  For me the love should start with attraction.i...     910"
      ]
     },
     "execution_count": 32,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "#max length message is of 910 words, let's print it\n",
    "messages[messages['length'] == 910]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 33,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "\"For me the love should start with attraction.i should feel that I need her every time around me.she should be the first thing which comes in my thoughts.I would start the day and end it with her.she should be there every time I dream.love will be then when my every breath has her name.my life should happen around her.my life will be named to her.I would cry for her.will give all my happiness and take all her sorrows.I will be ready to fight with anyone for her.I will be in love when I will be doing the craziest things for her.love will be when I don't have to proove anyone that my girl is the most beautiful lady on the whole planet.I will always be singing praises for her.love will be when I start up making chicken curry and end up makiing sambar.life will be the most beautiful then.will get every morning and thank god for the day because she is with me.I would like to say a lot..will tell later..\""
      ]
     },
     "execution_count": 33,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "messages[messages['length'] == 910]['messages'].iloc[0]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 35,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([<matplotlib.axes._subplots.AxesSubplot object at 0x11801570>,\n",
       "       <matplotlib.axes._subplots.AxesSubplot object at 0x11C81B30>],\n",
       "      dtype=object)"
      ]
     },
     "execution_count": 35,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 864x432 with 2 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "messages.hist(column='length', by='label', bins=75, figsize=(12,6))"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## From previous plots ,we come to a conclusion that spam messages have greater word counts in avg. than the ham messages word count"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Text Preprocessing\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 36,
   "metadata": {},
   "outputs": [],
   "source": [
    "import string"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 37,
   "metadata": {},
   "outputs": [],
   "source": [
    "mess = 'Sample message! Notice: it has punctuation.'"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 39,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "'!\"#$%&\\'()*+,-./:;<=>?@[\\\\]^_`{|}~'"
      ]
     },
     "execution_count": 39,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "string.punctuation"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 40,
   "metadata": {},
   "outputs": [],
   "source": [
    "nopunc = [c for c in mess if c not in string.punctuation]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 46,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "['S',\n",
       " 'a',\n",
       " 'm',\n",
       " 'p',\n",
       " 'l',\n",
       " 'e',\n",
       " ' ',\n",
       " 'm',\n",
       " 'e',\n",
       " 's',\n",
       " 's',\n",
       " 'a',\n",
       " 'g',\n",
       " 'e',\n",
       " ' ',\n",
       " 'N',\n",
       " 'o',\n",
       " 't',\n",
       " 'i',\n",
       " 'c',\n",
       " 'e',\n",
       " ' ',\n",
       " 'i',\n",
       " 't',\n",
       " ' ',\n",
       " 'h',\n",
       " 'a',\n",
       " 's',\n",
       " ' ',\n",
       " 'p',\n",
       " 'u',\n",
       " 'n',\n",
       " 'c',\n",
       " 't',\n",
       " 'u',\n",
       " 'a',\n",
       " 't',\n",
       " 'i',\n",
       " 'o',\n",
       " 'n']"
      ]
     },
     "execution_count": 46,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "nopunc"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## printing the stopwords"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 47,
   "metadata": {},
   "outputs": [],
   "source": [
    "from nltk.corpus import stopwords"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 48,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "['i',\n",
       " 'me',\n",
       " 'my',\n",
       " 'myself',\n",
       " 'we',\n",
       " 'our',\n",
       " 'ours',\n",
       " 'ourselves',\n",
       " 'you',\n",
       " \"you're\",\n",
       " \"you've\",\n",
       " \"you'll\",\n",
       " \"you'd\",\n",
       " 'your',\n",
       " 'yours',\n",
       " 'yourself',\n",
       " 'yourselves',\n",
       " 'he',\n",
       " 'him',\n",
       " 'his',\n",
       " 'himself',\n",
       " 'she',\n",
       " \"she's\",\n",
       " 'her',\n",
       " 'hers',\n",
       " 'herself',\n",
       " 'it',\n",
       " \"it's\",\n",
       " 'its',\n",
       " 'itself',\n",
       " 'they',\n",
       " 'them',\n",
       " 'their',\n",
       " 'theirs',\n",
       " 'themselves',\n",
       " 'what',\n",
       " 'which',\n",
       " 'who',\n",
       " 'whom',\n",
       " 'this',\n",
       " 'that',\n",
       " \"that'll\",\n",
       " 'these',\n",
       " 'those',\n",
       " 'am',\n",
       " 'is',\n",
       " 'are',\n",
       " 'was',\n",
       " 'were',\n",
       " 'be',\n",
       " 'been',\n",
       " 'being',\n",
       " 'have',\n",
       " 'has',\n",
       " 'had',\n",
       " 'having',\n",
       " 'do',\n",
       " 'does',\n",
       " 'did',\n",
       " 'doing',\n",
       " 'a',\n",
       " 'an',\n",
       " 'the',\n",
       " 'and',\n",
       " 'but',\n",
       " 'if',\n",
       " 'or',\n",
       " 'because',\n",
       " 'as',\n",
       " 'until',\n",
       " 'while',\n",
       " 'of',\n",
       " 'at',\n",
       " 'by',\n",
       " 'for',\n",
       " 'with',\n",
       " 'about',\n",
       " 'against',\n",
       " 'between',\n",
       " 'into',\n",
       " 'through',\n",
       " 'during',\n",
       " 'before',\n",
       " 'after',\n",
       " 'above',\n",
       " 'below',\n",
       " 'to',\n",
       " 'from',\n",
       " 'up',\n",
       " 'down',\n",
       " 'in',\n",
       " 'out',\n",
       " 'on',\n",
       " 'off',\n",
       " 'over',\n",
       " 'under',\n",
       " 'again',\n",
       " 'further',\n",
       " 'then',\n",
       " 'once',\n",
       " 'here',\n",
       " 'there',\n",
       " 'when',\n",
       " 'where',\n",
       " 'why',\n",
       " 'how',\n",
       " 'all',\n",
       " 'any',\n",
       " 'both',\n",
       " 'each',\n",
       " 'few',\n",
       " 'more',\n",
       " 'most',\n",
       " 'other',\n",
       " 'some',\n",
       " 'such',\n",
       " 'no',\n",
       " 'nor',\n",
       " 'not',\n",
       " 'only',\n",
       " 'own',\n",
       " 'same',\n",
       " 'so',\n",
       " 'than',\n",
       " 'too',\n",
       " 'very',\n",
       " 's',\n",
       " 't',\n",
       " 'can',\n",
       " 'will',\n",
       " 'just',\n",
       " 'don',\n",
       " \"don't\",\n",
       " 'should',\n",
       " \"should've\",\n",
       " 'now',\n",
       " 'd',\n",
       " 'll',\n",
       " 'm',\n",
       " 'o',\n",
       " 're',\n",
       " 've',\n",
       " 'y',\n",
       " 'ain',\n",
       " 'aren',\n",
       " \"aren't\",\n",
       " 'couldn',\n",
       " \"couldn't\",\n",
       " 'didn',\n",
       " \"didn't\",\n",
       " 'doesn',\n",
       " \"doesn't\",\n",
       " 'hadn',\n",
       " \"hadn't\",\n",
       " 'hasn',\n",
       " \"hasn't\",\n",
       " 'haven',\n",
       " \"haven't\",\n",
       " 'isn',\n",
       " \"isn't\",\n",
       " 'ma',\n",
       " 'mightn',\n",
       " \"mightn't\",\n",
       " 'mustn',\n",
       " \"mustn't\",\n",
       " 'needn',\n",
       " \"needn't\",\n",
       " 'shan',\n",
       " \"shan't\",\n",
       " 'shouldn',\n",
       " \"shouldn't\",\n",
       " 'wasn',\n",
       " \"wasn't\",\n",
       " 'weren',\n",
       " \"weren't\",\n",
       " 'won',\n",
       " \"won't\",\n",
       " 'wouldn',\n",
       " \"wouldn't\"]"
      ]
     },
     "execution_count": 48,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "stopwords.words('english')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 49,
   "metadata": {},
   "outputs": [],
   "source": [
    " nopunc = ''.join(nopunc)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 50,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "'Sample message Notice it has punctuation'"
      ]
     },
     "execution_count": 50,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "nopunc"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 56,
   "metadata": {},
   "outputs": [],
   "source": [
    "#x = ['a','b','c']"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 57,
   "metadata": {},
   "outputs": [],
   "source": [
    "#'-'.join(x)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 58,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "['Sample', 'message', 'Notice', 'it', 'has', 'punctuation']"
      ]
     },
     "execution_count": 58,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "nopunc.split()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 60,
   "metadata": {},
   "outputs": [],
   "source": [
    "clean_mess = [word for word in nopunc.split() if word.lower() not in stopwords.words('english')]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 61,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "['Sample', 'message', 'Notice', 'punctuation']"
      ]
     },
     "execution_count": 61,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "clean_mess"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 62,
   "metadata": {},
   "outputs": [],
   "source": [
    "def text_process(mess):\n",
    "    '''\n",
    "    1.remove punc\n",
    "    2.remove stop words\n",
    "    3.return list of clean text words\n",
    "    \n",
    "    '''\n",
    "    \n",
    "    nopunc = [char for char in mess if char not in string.punctuation]\n",
    "    \n",
    "    nopunc = ''.join(nopunc)\n",
    "    \n",
    "    return [word for word in nopunc.split() if word.lower() not in stopwords.words('english')]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 63,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>label</th>\n",
       "      <th>messages</th>\n",
       "      <th>length</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>ham</td>\n",
       "      <td>Go until jurong point, crazy.. Available only ...</td>\n",
       "      <td>111</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>ham</td>\n",
       "      <td>Ok lar... Joking wif u oni...</td>\n",
       "      <td>29</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>spam</td>\n",
       "      <td>Free entry in 2 a wkly comp to win FA Cup fina...</td>\n",
       "      <td>155</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>ham</td>\n",
       "      <td>U dun say so early hor... U c already then say...</td>\n",
       "      <td>49</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>ham</td>\n",
       "      <td>Nah I don't think he goes to usf, he lives aro...</td>\n",
       "      <td>61</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "  label                                           messages  length\n",
       "0   ham  Go until jurong point, crazy.. Available only ...     111\n",
       "1   ham                      Ok lar... Joking wif u oni...      29\n",
       "2  spam  Free entry in 2 a wkly comp to win FA Cup fina...     155\n",
       "3   ham  U dun say so early hor... U c already then say...      49\n",
       "4   ham  Nah I don't think he goes to usf, he lives aro...      61"
      ]
     },
     "execution_count": 63,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "messages.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 91,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "0    [Go, jurong, point, crazy, Available, bugis, n...\n",
       "1                       [Ok, lar, Joking, wif, u, oni]\n",
       "2    [Free, entry, 2, wkly, comp, win, FA, Cup, fin...\n",
       "3        [U, dun, say, early, hor, U, c, already, say]\n",
       "4    [Nah, dont, think, goes, usf, lives, around, t...\n",
       "Name: messages, dtype: object"
      ]
     },
     "execution_count": 91,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "messages['messages'].head(5).apply(text_process)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Count Vectorizer"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 66,
   "metadata": {},
   "outputs": [],
   "source": [
    "from sklearn.feature_extraction.text import CountVectorizer"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 67,
   "metadata": {},
   "outputs": [],
   "source": [
    "bow_transformer = CountVectorizer(analyzer=text_process)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 69,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "CountVectorizer(analyzer=<function text_process at 0x11B9F7C8>, binary=False,\n",
       "        decode_error='strict', dtype=<class 'numpy.int64'>,\n",
       "        encoding='utf-8', input='content', lowercase=True, max_df=1.0,\n",
       "        max_features=None, min_df=1, ngram_range=(1, 1), preprocessor=None,\n",
       "        stop_words=None, strip_accents=None,\n",
       "        token_pattern='(?u)\\\\b\\\\w\\\\w+\\\\b', tokenizer=None, vocabulary=None)"
      ]
     },
     "execution_count": 69,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "bow_transformer.fit(messages['messages'])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 70,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "11425\n"
     ]
    }
   ],
   "source": [
    "print(len(bow_transformer.vocabulary_))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 73,
   "metadata": {},
   "outputs": [],
   "source": [
    "mess4 = messages['messages'][3]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 75,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "U dun say so early hor... U c already then say...\n"
     ]
    }
   ],
   "source": [
    "print(mess4)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 76,
   "metadata": {},
   "outputs": [],
   "source": [
    "bow4 = bow_transformer.transform([mess4])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 77,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "  (0, 4068)\t2\n",
      "  (0, 4629)\t1\n",
      "  (0, 5261)\t1\n",
      "  (0, 6204)\t1\n",
      "  (0, 6222)\t1\n",
      "  (0, 7186)\t1\n",
      "  (0, 9554)\t2\n"
     ]
    }
   ],
   "source": [
    "print(bow4)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 78,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "(1, 11425)"
      ]
     },
     "execution_count": 78,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "bow4.shape"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 79,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "'U'"
      ]
     },
     "execution_count": 79,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "bow_transformer.get_feature_names()[4068]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 80,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "'say'"
      ]
     },
     "execution_count": 80,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "bow_transformer.get_feature_names()[9554]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 81,
   "metadata": {},
   "outputs": [],
   "source": [
    "messages_bow = bow_transformer.transform(messages['messages'])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 84,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Shape of Sparse Matrix:  (5572, 11425)\n",
      "Amount of Non-Zero occurences:  50548\n"
     ]
    }
   ],
   "source": [
    "print('Shape of Sparse Matrix: ', messages_bow.shape)\n",
    "print('Amount of Non-Zero occurences: ', messages_bow.nnz)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 86,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "sparsity: 0.07940295412668218\n"
     ]
    }
   ],
   "source": [
    "#to calculate the sparsity\n",
    "#dont need to learn the formula\n",
    "sparsity = (100.0 * messages_bow.nnz / (messages_bow.shape[0] * messages_bow.shape[1]))\n",
    "print('sparsity: {}'.format((sparsity)))"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# TF-iDF"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 87,
   "metadata": {},
   "outputs": [],
   "source": [
    "from sklearn.feature_extraction.text import TfidfTransformer"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 88,
   "metadata": {},
   "outputs": [],
   "source": [
    "tfidf_trans = TfidfTransformer().fit(messages_bow)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 89,
   "metadata": {},
   "outputs": [],
   "source": [
    "tfidf4 = tfidf_trans.transform(bow4)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 90,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "  (0, 9554)\t0.5385626262927564\n",
      "  (0, 7186)\t0.4389365653379857\n",
      "  (0, 6222)\t0.3187216892949149\n",
      "  (0, 6204)\t0.29953799723697416\n",
      "  (0, 5261)\t0.29729957405868723\n",
      "  (0, 4629)\t0.26619801906087187\n",
      "  (0, 4068)\t0.40832589933384067\n"
     ]
    }
   ],
   "source": [
    "print(tfidf4)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 93,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "8.527076498901426"
      ]
     },
     "execution_count": 93,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "#how to check the inverse document frequency of a particular word\n",
    "tfidf_trans.idf_[bow_transformer.vocabulary_['university']]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 103,
   "metadata": {},
   "outputs": [],
   "source": [
    "#to check the inverse document frequency of multiple messages\n",
    "message_tfidf = tfidf_trans.transform(messages_bow)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Training a model using NaiveBayes classifier algorithm"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 95,
   "metadata": {},
   "outputs": [],
   "source": [
    "from sklearn.naive_bayes import MultinomialNB"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 97,
   "metadata": {},
   "outputs": [],
   "source": [
    "spam_detect_model = MultinomialNB().fit(message_tfidf, messages['label'])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 99,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "'ham'"
      ]
     },
     "execution_count": 99,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "#classify single object model\n",
    "spam_detect_model.predict(tfidf4)[0]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 101,
   "metadata": {},
   "outputs": [],
   "source": [
    "all_pred = spam_detect_model.predict(message_tfidf)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 107,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "predicted: ham\n",
      "expected: ham\n"
     ]
    }
   ],
   "source": [
    "print('predicted:', spam_detect_model.predict(tfidf4)[0])\n",
    "print('expected:', messages.label[0])"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Testing the model, model evaluation"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Now we want to determine how well our model will do overall on the entire dataset. Let's begin by getting all the predictions:"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 108,
   "metadata": {},
   "outputs": [],
   "source": [
    "from sklearn.cross_validation import train_test_split"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 111,
   "metadata": {},
   "outputs": [],
   "source": [
    "msg_train, msg_test, label_train, label_test = train_test_split(messages['messages'], messages['label'], test_size=0.3 )"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 112,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "2692                          Hey tmr meet at bugis 930 ?\n",
       "48      Yeah hopefully, if tyler can't do it I could m...\n",
       "2614    Thanks for sending this mental ability question..\n",
       "5233     Hey what how about your project. Started aha da.\n",
       "2882    Printer is cool. I mean groovy. Wine is groovying\n",
       "1769    How. Its a little difficult but its a simple w...\n",
       "5291      Xy trying smth now. U eat already? We havent...\n",
       "5147    Get your garden ready for summer with a FREE s...\n",
       "2423            A bloo bloo bloo I'll miss the first bowl\n",
       "4162    Had your mobile 11 months or more? U R entitle...\n",
       "2371    That day ü say ü cut ur hair at paragon, is it...\n",
       "947     Ur cash-balance is currently 500 pounds - to m...\n",
       "967     I am not sure about night menu. . . I know onl...\n",
       "1765    Hi 07734396839 IBH Customer Loyalty Offer: The...\n",
       "3346        Reverse is cheating. That is not mathematics.\n",
       "4764                           Prepare to be pleasured :)\n",
       "1468           I wont touch you with out your permission.\n",
       "5378    Free entry to the gr8prizes wkly comp 4 a chan...\n",
       "4060    Moby Pub Quiz.Win a £100 High Street prize if ...\n",
       "1054       Jay's getting really impatient and belligerent\n",
       "2274                               Cold. Dont be sad dear\n",
       "2608    :-) yeah! Lol. Luckily i didn't have a starrin...\n",
       "2038                             Oh sorry please its over\n",
       "5102    This msg is for your mobile content order It h...\n",
       "2548    Text82228>> Get more ringtones, logos and game...\n",
       "3575                        Yeah sure I'll leave in a min\n",
       "3502    says the  &lt;#&gt;  year old with a man and m...\n",
       "396     From here after The performance award is calcu...\n",
       "2416    Huh means computational science... Y they like...\n",
       "1237                             How much are we getting?\n",
       "                              ...                        \n",
       "57                     Sorry, I'll call later in meeting.\n",
       "2735     Can you do a mag meeting this avo at some point?\n",
       "605                                Meet after lunch la...\n",
       "339                                Sorry, I'll call later\n",
       "2903    Bill, as in: Are there any letters for me. i’m...\n",
       "1433               Thanks for ve lovely wisheds. You rock\n",
       "4067    Fyi I'm gonna call you sporadically starting a...\n",
       "1599                  Daddy will take good care of you :)\n",
       "4161    i felt so...not any conveying reason.. Ese he....\n",
       "4049    Lol or I could just starve and lose a pound by...\n",
       "2026    Yes obviously, but you are the eggs-pert and t...\n",
       "911     My love ! How come it took you so long to leav...\n",
       "2577                 In sch but neva mind u eat 1st lor..\n",
       "5005    There's someone here that has a year  &lt;#&gt...\n",
       "1050    18 days to Euro2004 kickoff! U will be kept in...\n",
       "677     Maybe?! Say hi to  and find out if  got his ca...\n",
       "4449                            I sent them. Do you like?\n",
       "60      Your gonna have to pick up a $1 burger for you...\n",
       "1608    Jus telling u dat i'll b leaving 4 shanghai on...\n",
       "4535                      I have no money 4 steve mate! !\n",
       "3151          Yo! Howz u? girls never rang after india. L\n",
       "1205    WIN a year supply of CDs 4 a store of ur choic...\n",
       "1175    Yay! You better not have told that to 5 other ...\n",
       "1256    Just wait till end of march when el nino gets ...\n",
       "702                                Sorry, I'll call later\n",
       "5297    My darling sister. How are you doing. When's s...\n",
       "5361           Yep get with the program. You're slacking.\n",
       "2704    Yup no more already... Thanx 4 printing n hand...\n",
       "4749    The beauty of life is in next second.. which h...\n",
       "550     Ok give me 5 minutes I think I see her. BTW yo...\n",
       "Name: messages, Length: 3900, dtype: object"
      ]
     },
     "execution_count": 112,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "msg_train"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Pipelining"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 113,
   "metadata": {},
   "outputs": [],
   "source": [
    "from sklearn.pipeline import Pipeline"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 125,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "\"\\nfrom sklearn.pipeline import Pipeline\\n\\npipeline = Pipeline([\\n    ('bow', CountVectorizer(analyzer=text_process)),  # strings to token integer counts\\n    ('tfidf', TfidfTransformer()),  # integer counts to weighted TF-IDF scores\\n    ('classifier', MultinomialNB()),  # train on TF-IDF vectors w/ Naive Bayes classifier\\n])\""
      ]
     },
     "execution_count": 125,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "'''\n",
    "from sklearn.pipeline import Pipeline\n",
    "\n",
    "pipeline = Pipeline([\n",
    "    ('bow', CountVectorizer(analyzer=text_process)),  # strings to token integer counts\n",
    "    ('tfidf', TfidfTransformer()),  # integer counts to weighted TF-IDF scores\n",
    "    ('classifier', MultinomialNB()),  # train on TF-IDF vectors w/ Naive Bayes classifier\n",
    "])'''"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 126,
   "metadata": {},
   "outputs": [],
   "source": [
    "from sklearn.ensemble import RandomForestClassifier"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 127,
   "metadata": {},
   "outputs": [],
   "source": [
    "from sklearn.pipeline import Pipeline\n",
    "\n",
    "pipeline = Pipeline([\n",
    "    ('bow', CountVectorizer(analyzer=text_process)),  # strings to token integer counts\n",
    "    ('tfidf', TfidfTransformer()),  # integer counts to weighted TF-IDF scores\n",
    "    ('classifier', RandomForestClassifier()),  # train on TF-IDF vectors w/ Naive Bayes classifier\n",
    "])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 128,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "Pipeline(memory=None,\n",
       "     steps=[('bow', CountVectorizer(analyzer=<function text_process at 0x11B9F7C8>, binary=False,\n",
       "        decode_error='strict', dtype=<class 'numpy.int64'>,\n",
       "        encoding='utf-8', input='content', lowercase=True, max_df=1.0,\n",
       "        max_features=None, min_df=1, ngram_range=(1, 1), preprocessor=None,\n",
       "...n_jobs=1,\n",
       "            oob_score=False, random_state=None, verbose=0,\n",
       "            warm_start=False))])"
      ]
     },
     "execution_count": 128,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "pipeline.fit(msg_train, label_train)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 129,
   "metadata": {},
   "outputs": [],
   "source": [
    "preds = pipeline.predict(msg_test)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## classification report and confusion matrix"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 130,
   "metadata": {},
   "outputs": [],
   "source": [
    "from sklearn.metrics import classification_report, confusion_matrix"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 132,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "             precision    recall  f1-score   support\n",
      "\n",
      "        ham       1.00      0.96      0.98      1529\n",
      "       spam       0.72      0.99      0.84       143\n",
      "\n",
      "avg / total       0.98      0.97      0.97      1672\n",
      "\n",
      "\n",
      "\n",
      "\n",
      "[[1474   55]\n",
      " [   1  142]]\n"
     ]
    }
   ],
   "source": [
    "print(classification_report(preds, label_test))\n",
    "print('\\n\\n')\n",
    "print(confusion_matrix(preds, label_test))"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Good Job!!"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.6.5"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}
