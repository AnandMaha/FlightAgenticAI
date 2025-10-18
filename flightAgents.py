import openai
import json
from datetime import datetime, timedelta
from typing import List, Dict, Any

# Set your OpenAI API key
openai.api_key = ""

class ChatGPTPoweredFlightAgent:
    """Agentic AI system powered entirely by ChatGPT API"""
    
    def __init__(self):
        self.client = openai.OpenAI(api_key=openai.api_key)
        self.conversation_history = []
    
    def add_to_history(self, role: str, content: str):
        """Add message to conversation history"""
        self.conversation_history.append({"role": role, "content": content})
    
    def chat_completion(self, prompt: str, system_message: str = None) -> str:
        """Make ChatGPT API call"""
        messages = []
        
        if system_message:
            messages.append({"role": "system", "content": system_message})
        
        # Add conversation history
        messages.extend(self.conversation_history)
        
        # Add current prompt
        messages.append({"role": "user", "content": prompt})
        
        try:
            response = self.client.chat.completions.create(
                model="gpt-4",
                messages=messages,
                temperature=0.7,
                max_tokens=2000
            )
            
            response_content = response.choices[0].message.content
            self.add_to_history("user", prompt)
            self.add_to_history("assistant", response_content)
            
            return response_content
            
        except Exception as e:
            return f"Error: {str(e)}"
    
    def get_weather_analysis(self, city: str, start_date: str, end_date: str, temp_range: tuple) -> str:
        """Use ChatGPT to analyze weather patterns"""
        
        system_msg = """You are a weather analysis expert. Analyze historical weather patterns 
        and provide realistic weather forecasts based on seasonal data."""
        
        prompt = f"""
        Analyze the typical weather patterns for {city} between {start_date} and {end_date}.
        
        Based on historical seasonal data:
        1. Predict which dates are most likely to have temperatures between {temp_range[0]}°F and {temp_range[1]}°F
        2. Provide a list of specific dates that match this temperature range
        3. Include the expected temperature and conditions for each date
        
        Return your response in this JSON format:
        {{
            "ideal_dates": [
                {{
                    "date": "YYYY-MM-DD",
                    "expected_temp": 68,
                    "conditions": "Partly cloudy"
                }}
            ],
            "analysis": "Brief analysis of weather patterns"
        }}
        """
        
        response = self.chat_completion(prompt, system_msg)
        return response
    
    def get_airline_rankings(self) -> str:
        """Use ChatGPT to get airline reviews and rankings"""
        
        system_msg = """You are an airline industry expert with knowledge of recent customer reviews, 
        on-time performance, and service quality for major US airlines."""
        
        prompt = """
        Based on recent customer reviews, on-time performance, and service quality for 2024, 
        rank the top US airlines for transcontinental flights (like SFO to New York).
        
        Consider factors like:
        - Customer satisfaction
        - On-time performance
        - Seat comfort
        - Customer service
        - Baggage handling
        - In-flight amenities
        
        Return your top 2 airlines with their ratings and brief justification.
        
        Return in this JSON format:
        {
            "top_airlines": [
                {
                    "name": "Airline Name",
                    "rating": 4.5,
                    "justification": "Brief reason for ranking"
                }
            ]
        }
        """
        
        response = self.chat_completion(prompt, system_msg)
        return response
    
    def search_flights(self, origin: str, destination: str, ideal_dates: list, airlines: list) -> str:
        """Use ChatGPT to generate realistic flight options"""
        
        system_msg = """You are a flight search expert. Generate realistic flight options 
        including pricing, times, and flight numbers that match real-world patterns."""
        
        prompt = f"""
        Generate realistic round-trip flight options from {origin} to {destination}.
        
        Requirements:
        - Origin: {origin}
        - Destination: {destination}
        - Round-trip flights only
        - Departure dates: {ideal_dates}
        - Return dates: 7 days after departure
        - Preferred airlines: {', '.join(airlines)}
        - Include realistic pricing, flight times, and flight numbers
        
        Generate 5-8 flight options with varying:
        - Departure times (morning, afternoon, evening)
        - Prices (realistic range for this route)
        - Flight durations
        - Non-stop and 1-stop options
        
        Return in this JSON format:
        {{
            "flight_options": [
                {{
                    "airline": "Airline Name",
                    "outbound": {{
                        "flight_number": "DL123",
                        "departure_time": "08:00 AM",
                        "arrival_time": "04:30 PM",
                        "date": "2024-10-05",
                        "duration": "5h 30m",
                        "stops": 0
                    }},
                    "inbound": {{
                        "flight_number": "DL124",
                        "departure_time": "10:00 AM",
                        "arrival_time": "01:30 PM",
                        "date": "2024-10-12",
                        "duration": "6h 30m",
                        "stops": 0
                    }},
                    "total_price": 450.00,
                    "booking_link": "https://airline.com/booking/DL123-DL124"
                }}
            ]
        }}
        """
        
        response = self.chat_completion(prompt, system_msg)
        return response
    
    def find_optimal_flights(self, origin: str, destination: str, 
                           start_date: str, end_date: str, 
                           temp_min: int, temp_max: int) -> Dict[str, Any]:
        """Main method to coordinate the flight search using ChatGPT"""
        
        print("🤖 Starting AI-powered flight search...")
        print("=" * 50)
        
        # Step 1: Get airline rankings
        print("1. Analyzing airline reviews and rankings...")
        airlines_response = self.get_airline_rankings()
        airlines_data = self._parse_json_response(airlines_response)
        top_airlines = [airline["name"] for airline in airlines_data.get("top_airlines", [])]
        
        print(f"   ✅ Top airlines: {', '.join(top_airlines)}")
        
        # Step 2: Analyze weather patterns
        print("2. Analyzing weather patterns...")
        weather_response = self.get_weather_analysis(destination, start_date, end_date, (temp_min, temp_max))
        weather_data = self._parse_json_response(weather_response)
        ideal_dates = [date_info["date"] for date_info in weather_data.get("ideal_dates", [])]
        
        print(f"   ✅ Found {len(ideal_dates)} ideal weather dates")
        
        if not ideal_dates:
            return {"error": "No dates found with ideal weather conditions"}
        
        # Step 3: Search for flights
        print("3. Searching for flight options...")
        flights_response = self.search_flights(origin, destination, ideal_dates[:3], top_airlines)
        flights_data = self._parse_json_response(flights_response)
        
        print(f"   ✅ Found {len(flights_data.get('flight_options', []))} flight options")
        
        # Compile final results
        results = {
            "search_criteria": {
                "route": f"{origin} to {destination}",
                "dates": f"{start_date} to {end_date}",
                "temperature_range": f"{temp_min}-{temp_max}°F",
                "preferred_airlines": top_airlines
            },
            "weather_analysis": weather_data.get("analysis", ""),
            "ideal_dates": ideal_dates,
            "flight_options": flights_data.get("flight_options", []),
            "top_airlines": airlines_data.get("top_airlines", [])
        }
        
        return results
    
    def _parse_json_response(self, response: str) -> Dict[str, Any]:
        """Parse JSON response from ChatGPT, handling markdown code blocks"""
        try:
            # Remove markdown code blocks if present
            cleaned_response = response.replace('```json', '').replace('```', '').strip()
            return json.loads(cleaned_response)
        except json.JSONDecodeError:
            # If JSON parsing fails, return empty dict
            print(f"⚠️  JSON parsing failed for response: {response[:100]}...")
            return {}

class FlightSearchUI:
    """User interface for displaying results"""
    
    @staticmethod
    def display_results(results: Dict[str, Any]):
        """Display formatted results"""
        
        if "error" in results:
            print(f"❌ Error: {results['error']}")
            return
        
        print("\n" + "="*80)
        print("🎯 AI-POWERED FLIGHT SEARCH RESULTS")
        print("="*80)
        
        # Display search criteria
        criteria = results["search_criteria"]
        print(f"\n📋 SEARCH CRITERIA:")
        print(f"   • Route: {criteria['route']} (Round Trip)")
        print(f"   • Dates: {criteria['dates']}")
        print(f"   • Ideal Temperature: {criteria['temperature_range']}")
        print(f"   • Preferred Airlines: {', '.join(criteria['preferred_airlines'])}")
        
        # Display weather analysis
        print(f"\n🌤️  WEATHER ANALYSIS:")
        print(f"   {results['weather_analysis']}")
        print(f"   • Ideal dates found: {len(results['ideal_dates'])}")
        
        # Display airline rankings
        print(f"\n🏆 AIRLINE RANKINGS:")
        for airline in results["top_airlines"]:
            print(f"   • {airline['name']}: ⭐ {airline['rating']}/5 - {airline['justification']}")
        
        # Display flight options
        print(f"\n✈️  FLIGHT OPTIONS:")
        print("-" * 80)
        
        for i, option in enumerate(results["flight_options"], 1):
            print(f"\n{i}. {option['airline']} - ${option['total_price']}")
            print(f"   OUTBOUND: {option['outbound']['date']}")
            print(f"     Flight {option['outbound']['flight_number']}: {option['outbound']['departure_time']} → {option['outbound']['arrival_time']}")
            print(f"     Duration: {option['outbound']['duration']} • Stops: {option['outbound']['stops']}")
            
            print(f"   INBOUND: {option['inbound']['date']}")
            print(f"     Flight {option['inbound']['flight_number']}: {option['inbound']['departure_time']} → {option['inbound']['arrival_time']}")
            print(f"     Duration: {option['inbound']['duration']} • Stops: {option['inbound']['stops']}")
            
            print(f"   💰 Total Price: ${option['total_price']}")
            print(f"   🔗 {option.get('booking_link', 'Visit airline website to book')}")

def main():
    """Main function to run the ChatGPT-powered flight search"""
    
    # Initialize the AI agent
    agent = ChatGPTPoweredFlightAgent()
    
    # Define search parameters
    search_params = {
        "origin": "SFO",
        "destination": "New York",
        "start_date": "2024-10-01",
        "end_date": "2024-10-15",
        "temp_min": 65,
        "temp_max": 70
    }
    
    print("🚀 ChatGPT-Powered Agentic Flight Search")
    print("==========================================")
    
    # Execute the search
    results = agent.find_optimal_flights(**search_params)
    
    # Display results
    FlightSearchUI.display_results(results)
    
    # Save results to file
    with open("flight_search_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n💾 Results saved to 'flight_search_results.json'")

# Additional utility functions
def quick_search():
    """Quick search function for common routes"""
    agent = ChatGPTPoweredFlightAgent()
    
    # You can modify these parameters
    results = agent.find_optimal_flights(
        origin="SFO",
        destination="New York", 
        start_date="2024-10-01",
        end_date="2024-10-15",
        temp_min=65,
        temp_max=70
    )
    
    FlightSearchUI.display_results(results)

if __name__ == "__main__":
    main()
    
    # Uncomment for quick testing
    # quick_search()