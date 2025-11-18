import argparse
from neo4j import GraphDatabase
from datetime import datetime, timedelta
import math
import statistics

# Import centralized configuration
import config

import math
import statistics

def get_historical_conflict_stats(driver, actor1_name, actor2_name, event_root_code=None):
    """
    Queries the graph to get the historical baseline of daily conflict events
    between two actors. The calculation is done in Python to ensure compatibility.
    
    Returns a dictionary with mean and standard deviation.
    """
    print(f"Calculating historical baseline for interactions between {actor1_name} and {actor2_name}...")
    
    query = """
    MATCH (a:GeopoliticalActor {name: $actor1})-[r:INTERACTED_WITH]->(b:GeopoliticalActor {name: $actor2})
    WHERE r.goldstein_scale IS NOT NULL AND toFloat(r.goldstein_scale) < 0
    """
    
    if event_root_code:
        query += " AND r.root_code = $event_code"

    query += """
    WITH r.date AS event_date, count(r) AS daily_count
    RETURN daily_count
    """
    
    daily_counts = []
    with driver.session() as session:
        results = session.run(query, actor1=actor1_name, actor2=actor2_name, event_code=event_root_code)
        daily_counts = [record["daily_count"] for record in results]
        
    if daily_counts:
        mean = statistics.mean(daily_counts)
        # Use statistics.stdev if there's more than one data point, otherwise std_dev is 0
        std_dev = statistics.stdev(daily_counts) if len(daily_counts) > 1 else 0.0
        # Ensure std_dev is at least 1.0 to avoid division by zero issues for anomaly detection
        std_dev = max(std_dev, 1.0)

        stats = {
            "mean": mean,
            "std_dev": std_dev,
            "total_events": sum(daily_counts),
            "days_with_events": len(daily_counts)
        }
        print(f"  - Baseline: Mean {stats['mean']:.2f} conflict events/day, StdDev: {stats['std_dev']:.2f}")
        return stats
    else:
        print("  - No historical conflict data found for this actor pair.")
        return None

def get_recent_conflict_count(driver, actor1_name, actor2_name, days=1, event_root_code=None):
    """
    Queries the graph for the number of conflict events in the last N days.
    """
    print(f"Fetching recent conflict events (last {days} day(s))...")
    
    start_date = datetime.now() - timedelta(days=days)
    # Format as YYYYMMDD string for direct comparison with the 'sqldate' property
    start_date_str = start_date.strftime('%Y%m%d')

    query = """
    MATCH (a:GeopoliticalActor {name: $actor1})-[r:INTERACTED_WITH]->(b:GeopoliticalActor {name: $actor2})
    WHERE r.goldstein_scale IS NOT NULL AND toFloat(r.goldstein_scale) < 0 AND r.date >= $start_date
    """

    if event_root_code:
        query += " AND r.root_code = $event_code"
        
    query += " RETURN count(r) AS recent_count"

    with driver.session() as session:
        result = session.run(query, actor1=actor1_name, actor2=actor2_name, start_date=start_date_str, event_code=event_root_code).single()
        
    count = result['recent_count'] if result else 0
    print(f"  - Found {count} conflict events in the last {days} day(s).")
    return count

def main():
    parser = argparse.ArgumentParser(description="Graph-based Anomaly Detector for Geopolitical Events.")
    parser.add_argument("actor1", help="Name of the first geopolitical actor (e.g., 'UNITED STATES').")
    parser.add_argument("actor2", help="Name of the second geopolitical actor (e.g., 'CHINA').")
    parser.add_argument("--days", type=int, default=1, help="Number of recent days to check for anomalies.")
    parser.add_argument("--event-code", type=str, default=None, help="Optional: Specific event root code to filter by (e.g., '14' for protest).")
    parser.add_argument("--threshold", type=float, default=3.0, help="Z-score threshold for anomaly detection.")
    
    args = parser.parse_args()

    print(f"--- Starting Anomaly Detection for {args.actor1} <-> {args.actor2} ---")
    
    driver = GraphDatabase.driver(config.NEO4J_URI, auth=(config.NEO4J_USER, config.NEO4J_PASSWORD))

    try:
        # 1. Get historical baseline
        baseline_stats = get_historical_conflict_stats(driver, args.actor1, args.actor2, args.event_code)
        
        if not baseline_stats:
            return

        # 2. Get recent activity
        recent_count = get_recent_conflict_count(driver, args.actor1, args.actor2, args.days, args.event_code)
        
        # 3. Calculate Z-score and check for anomaly
        if baseline_stats['std_dev'] > 0:
            # We normalize the recent count to a per-day value for a fair comparison
            recent_rate = recent_count / args.days
            
            z_score = (recent_rate - baseline_stats['mean']) / baseline_stats['std_dev']
            
            print(f"\n--- Analysis ---")
            print(f"  - Recent Daily Rate: {recent_rate:.2f} events/day")
            print(f"  - Historical Daily Mean: {baseline_stats['mean']:.2f} events/day")
            print(f"  - Z-Score: {z_score:.2f}")

            if z_score > args.threshold:
                print(f"\n>> ALERT: Anomaly detected! Z-score ({z_score:.2f}) exceeds threshold of {args.threshold}. <<")
            else:
                print("\n>> No anomaly detected. Event rate is within normal parameters. <<")
        else:
            print("\n--- Analysis ---")
            print("  - Cannot calculate Z-score because historical standard deviation is zero.")

    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        driver.close()
        print("\n--- Detection Run Finished ---")

if __name__ == "__main__":
    main()
