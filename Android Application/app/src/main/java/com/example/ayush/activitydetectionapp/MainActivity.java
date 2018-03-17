package com.example.ayush.activitydetectionapp;

import android.content.Context;
import android.hardware.Sensor;
import android.hardware.SensorEvent;
import android.hardware.SensorEventListener;
import android.hardware.SensorManager;
import android.os.Bundle;
import android.os.Handler;
import android.os.Message;
import android.support.wearable.activity.WearableActivity;
import android.util.Log;
import android.view.View;
import android.widget.TextView;

import com.google.android.gms.wearable.WearableListenerService;
import com.loopj.android.http.AsyncHttpClient;
import com.loopj.android.http.SyncHttpClient;
import com.loopj.android.http.TextHttpResponseHandler;

import java.util.ArrayList;
import java.util.Date;
import java.util.List;
import java.util.Random;
import java.util.Timer;
import java.util.TimerTask;

import cz.msebera.android.httpclient.Header;
import cz.msebera.android.httpclient.entity.StringEntity;

public class MainActivity extends WearableActivity implements SensorEventListener{

    private TextView mTextView;
    private static final String TAG = "MainActivity";

    private static String accData = "";
    private static String gyroData = "";
    private static String magData = "";
    String resp = "";
    private static final String acc = "\"acceleration\"";
    private static final String gyro = "\"gyroscope\"";
    private static final String mag = "\"magnetic\"";
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        mTextView = (TextView) findViewById(R.id.text);
        // Enables Always-on
        setAmbientEnabled();
    }

    private void startCollecting() {
        SensorManager mSensorManager = ((SensorManager)getSystemService(SENSOR_SERVICE));
        Sensor mySensor = mSensorManager.getDefaultSensor(Sensor.TYPE_ACCELEROMETER);

        mSensorManager.registerListener(this,mySensor,SensorManager.SENSOR_DELAY_NORMAL);

    }

    public void onAccuracyChanged(Sensor sensor, int accuracy) {
        Log.d(TAG, "onAccuracyChanged - accuracy: " + accuracy);
    }

    public void onSensorChanged(SensorEvent event) {

        long timeInMillis = (new Date()).getTime()
                + (event.timestamp - System.nanoTime()) / 1000000L;

        float x = event.values[0];
        float y = event.values[1];
        float z = event.values[2];
        String myData = "[" + Long.toString(timeInMillis) + ", " + Float.toString(x)
                         + ", "+ Float.toString(y) + ", " + Float.toString(z)
                         + ", 0, \"unknown\"]";
        if (event.sensor.getType() == Sensor.TYPE_ACCELEROMETER) {
            if(accData.equals(""))
                accData = accData + myData;
            else
                accData = accData + ", " + myData;
            //mTextView.setText(msg);
            //Log.d(TAG, msg);
        }
        else if(event.sensor.getType() == Sensor.TYPE_MAGNETIC_FIELD){
            if(magData.equals(""))
                magData = magData + myData;
            else
                magData = magData + ", " + myData;
        }
        else if(event.sensor.getType() == Sensor.TYPE_GYROSCOPE){
            if(gyroData.equals(""))
                gyroData = gyroData + myData;
            else
                gyroData = gyroData + ", " + myData;
        }
        else
            Log.d(TAG, "Unknown sensor type");
    }

    public void callWebService(View view)
    {
        final String BASE_URL = "";//Replace with URL OF SERVER
        final String API_KEY = ""; //Replace with API KEY OF SERVER
        final SyncHttpClient client = new SyncHttpClient();
        client.addHeader("Content-Type", "application/json");
        client.addHeader("Authorization", "Bearer " + API_KEY);
        final Context context = this;
        final String myNum = "6.0";
        final long period = 6400;
        startCollecting();
        new Timer().schedule(new TimerTask() {
            @Override
            public void run() {
                cpost(client, context, BASE_URL);//,myNum);//, myNum);
                // do your task here
            }
        }, period, period);

    }
    public void cpost(SyncHttpClient client, Context context, String BASE_URL)//, String myNum)
    {
        try {

            //String interim = "[" + myNum + "]]}";

            String res = "{\"input_array\": {" + acc + ": [" + accData +
                         "]," + gyro + ": [" + gyroData +
                         "]," + mag + ": [" + magData +
                         "]}}";
            //String res = "{\"input_array\": [" + interim;
            StringEntity entity = new StringEntity(res);
            client.post(context, BASE_URL, entity, "application/json",

                    new TextHttpResponseHandler() {
                        @Override
                        public void onFailure(int statusCode, Header[] headers, String responseString, Throwable throwable) {
                            resp = responseString;
                            mHandler.obtainMessage(1).sendToTarget();
                            //mTextView = (TextView) findViewById(R.id.text);
                            //mTextView.setText(resp.substring(2,resp.length()-2));
                        }

                        @Override
                        public void onSuccess(int statusCode, Header[] headers, String responseString) {
                            resp = responseString;

                            mHandler.obtainMessage(1).sendToTarget();
                            //mTextView = (TextView) findViewById(R.id.text);
                            //mTextView.setText(resp.substring(2,resp.length()-2));
                        }
                    });

        }catch (Exception e){}
        accData = "";
        gyroData = "";
        magData = "";
    }
    public Handler mHandler = new Handler() {
        public void handleMessage(Message msg) {
            mTextView = (TextView) findViewById(R.id.text);
            mTextView.setText(resp); //this is the textview
        }
    };
}
