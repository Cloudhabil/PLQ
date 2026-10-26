import sqlite3
import os
import sys
from typing import Tuple
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "topos-lang", "src")))
from topos.core.clifford import PGA301Motor, Multivector

def main():
    # 1. Initialize the PGA301Motor
    # Rotation: 90 degrees (pi/2 radians) around Z-axis (0, 0, 1)
    # Translation: (1, 2, 0)
    axis = (0.0, 0.0, 1.0)
    angle = 90.0 * (3.141592653589793 / 180.0)
    translation = (1.0, 2.0, 0.0)

    # Create the motor
    motor = PGA301Motor.from_axis_angle_translation(
        axis=axis, 
        angle=angle, 
        translation=translation
    )

    print(f"Motor: {motor.multivector}")
    print(f"Is Unit Motor: {motor.is_unit_motor()}")

    # 2. Create/Populate the database
    # We'll create a database file and insert the motor's components
    db_path = "motor_results.db"
    
    # Ensure the directory exists (it's in the current working directory)
    # If we want it in a specific folder, we'd use that path.
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Create table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS motors (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            axis TEXT,
            angle REAL,
            translation TEXT,
            motor_components TEXT,
            is_unit REAL
        )
    """)
    
    # Insert data
    cursor.execute("""
        INSERT INTO motors (axis, angle, translation, motor_components, is_unit)
        VALUES (?, ?, ?, ?, ?)
    """, (
        str(axis),
        angle,
        str(translation),
        str(motor.multivector.components),
        1.0 if motor.is_unit_motor() else 0.0
    ))
    
    conn.commit()
    conn.close()
    
    print(f"Successfully populated {db_path}")

if __name__ == "__main__":
    main()
