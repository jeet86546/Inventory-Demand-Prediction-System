{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 2,
   "id": "6642b862-b0f1-434d-b9cf-6196d82dd672",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "🟢 Live Data Generator Started\n",
      "✅ Added New Live Record\n",
      "✅ Added New Live Record\n",
      "✅ Added New Live Record\n",
      "✅ Added New Live Record\n"
     ]
    },
    {
     "ename": "KeyboardInterrupt",
     "evalue": "",
     "output_type": "error",
     "traceback": [
      "\u001b[1;31m---------------------------------------------------------------------------\u001b[0m",
      "\u001b[1;31mKeyboardInterrupt\u001b[0m                         Traceback (most recent call last)",
      "Cell \u001b[1;32mIn[2], line 56\u001b[0m\n\u001b[0;32m     53\u001b[0m \u001b[38;5;28mprint\u001b[39m(\u001b[38;5;124m\"\u001b[39m\u001b[38;5;124m✅ Added New Live Record\u001b[39m\u001b[38;5;124m\"\u001b[39m)\n\u001b[0;32m     55\u001b[0m \u001b[38;5;66;03m# Wait 5 seconds\u001b[39;00m\n\u001b[1;32m---> 56\u001b[0m time\u001b[38;5;241m.\u001b[39msleep(\u001b[38;5;241m5\u001b[39m)\n",
      "\u001b[1;31mKeyboardInterrupt\u001b[0m: "
     ]
    }
   ],
   "source": [
    "import pandas as pd\n",
    "import numpy as np\n",
    "import time\n",
    "from datetime import datetime, timedelta\n",
    "\n",
    "# Load original dataset\n",
    "df = pd.read_csv(\"walmart_cleaned.csv\", parse_dates=[\"Date\"])\n",
    "\n",
    "LIVE_FILE = \"live_sales.csv\"\n",
    "\n",
    "print(\"🟢 Live Data Generator Started\")\n",
    "\n",
    "while True:\n",
    "\n",
    "    # Pick random row pattern\n",
    "    sample = df.sample(1).iloc[0].copy()\n",
    "\n",
    "    # Generate new date\n",
    "    sample[\"Date\"] = datetime.now()\n",
    "\n",
    "    # Add realistic sales variation\n",
    "    variation = np.random.uniform(0.9, 1.1)\n",
    "\n",
    "    sample[\"Weekly_Sales\"] = round(\n",
    "        sample[\"Weekly_Sales\"] * variation,\n",
    "        2\n",
    "    )\n",
    "\n",
    "    # Slight temperature variation\n",
    "    sample[\"Temperature\"] += np.random.uniform(-2, 2)\n",
    "\n",
    "    # Slight fuel price variation\n",
    "    sample[\"Fuel_Price\"] += np.random.uniform(-0.1, 0.1)\n",
    "\n",
    "    # Slight CPI variation\n",
    "    sample[\"CPI\"] += np.random.uniform(-1, 1)\n",
    "\n",
    "    # Slight unemployment variation\n",
    "    sample[\"Unemployment\"] += np.random.uniform(-0.2, 0.2)\n",
    "\n",
    "    # Convert to dataframe\n",
    "    new_row = pd.DataFrame([sample])\n",
    "\n",
    "    # Append to live csv\n",
    "    new_row.to_csv(\n",
    "        LIVE_FILE,\n",
    "        mode='a',\n",
    "        header=False,\n",
    "        index=False\n",
    "    )\n",
    "\n",
    "    print(\"✅ Added New Live Record\")\n",
    "\n",
    "    # Wait 5 seconds\n",
    "    time.sleep(5)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "b074c6e5-82d0-4711-bbc6-86e82f606405",
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "d682f78c-4db8-47bb-9eb0-39f37ceae915",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
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
   "version": "3.13.5"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
