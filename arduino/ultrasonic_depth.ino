#define TRIG 9 // TRIG 핀 설정 (초음파 보내는 핀)
#define ECHO 8 // ECHO 핀 설정 (초음파 받는 핀)
#define GREEN 6
#define RED 7
void setup()
{
  Serial.begin(9600);
  pinMode(TRIG, OUTPUT);
  pinMode(ECHO, INPUT);
  pinMode(GREEN, OUTPUT);
  pinMode(RED, OUTPUT);
}

void loop()
{
  long duration, distance;
  digitalWrite(TRIG, LOW);
  delayMicroseconds(2);
  digitalWrite(TRIG, HIGH);
  delayMicroseconds(10);
  digitalWrite(TRIG, LOW);
  duration = pulseIn(ECHO, HIGH);
  distance = duration * 17 / 1000;
  Serial.println(duration);
  Serial.print("\nDIstance : ");
  Serial.print(distance);
  Serial.println(" Cm");

  if (distance >= 10 && distance <= 11)
  {
    digitalWrite(GREEN, HIGH);
    digitalWrite(RED, LOW);
  }
  else
  {
    digitalWrite(GREEN, LOW);
    digitalWrite(RED, HIGH);
  }
  delay(1000);
}