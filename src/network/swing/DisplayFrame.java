package network.swing;

import java.awt.Color;
import java.awt.Dimension;
import java.awt.Font;
import java.awt.Graphics;
import java.awt.Graphics2D;
import java.awt.RenderingHints;
import java.awt.event.MouseEvent;
import java.awt.event.MouseMotionAdapter;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.util.Arrays;
import javax.swing.BorderFactory;
import javax.swing.Box;
import javax.swing.BoxLayout;
import javax.swing.JButton;
import javax.swing.JFileChooser;
import javax.swing.JFrame;
import javax.swing.JLabel;
import javax.swing.JPanel;
import network.App;
import network.trainer.NetworkTrainer;

public class DisplayFrame extends JFrame {

  private static final byte CELL_SIZE = 20;
  private static final short SIDE_LENGTH = CELL_SIZE * App.IMAGE_SIDE;
  private static final Color COLOUR = new Color(50, 50, 80);

  private final double[][] image = new double[App.IMAGE_SIDE][App.IMAGE_SIDE];
  private double[] results = new double[10];
  private final DrawPane draw;
  private final StatsPane stats;
  private final transient CustomMouseAdapter mouseAdapter;

  private class CustomMouseAdapter extends MouseMotionAdapter {

    private NetworkTrainer<Byte> trainer;

    @Override
    public void mouseDragged(MouseEvent e) {
      if (trainer == null) return;

      final int x = e.getX() / CELL_SIZE;
      final int y = e.getY() / CELL_SIZE;

      if (x < 0 || y < 0 || x >= App.IMAGE_SIDE || y >= App.IMAGE_SIDE) return;

      image[x][y] = 1;

      draw.repaint();
      results =
        trainer.testImage(
          Arrays.stream(image).flatMapToDouble(Arrays::stream).toArray()
        );
      stats.update();
    }
  }

  private class DrawPane extends JPanel {

    DrawPane() {
      final Dimension size = new Dimension(SIDE_LENGTH, SIDE_LENGTH);
      setPreferredSize(size);
      setMaximumSize(size);
      setMinimumSize(size);
      setBorder(BorderFactory.createLineBorder(Color.WHITE));
      setBackground(new Color(30, 30, 60));
      addMouseMotionListener(mouseAdapter);
    }

    @Override
    protected void paintComponent(Graphics g) {
      super.paintComponent(g);

      final Graphics2D g2 = (Graphics2D) g;
      g2.setRenderingHint(
        RenderingHints.KEY_ANTIALIASING,
        RenderingHints.VALUE_ANTIALIAS_ON
      );
      g2.setRenderingHint(
        RenderingHints.KEY_ALPHA_INTERPOLATION,
        RenderingHints.VALUE_ALPHA_INTERPOLATION_QUALITY
      );

      for (int x = 0; x < image.length; x++) {
        for (int y = 0; y < image.length; y++) {
          final int scaledX = x * CELL_SIZE;
          final int scaledY = y * CELL_SIZE;
          g2.setColor(new Color(255, 255, 255, (int) (image[x][y] * 255)));
          g2.fillRect(scaledX, scaledY, CELL_SIZE + 1, CELL_SIZE + 1);
        }
      }
    }
  }

  private class StatsPane extends JPanel {

    private class Label extends JLabel {

      Label(String label, Font font) {
        super(label);
        setFont(font);
        setForeground(Color.WHITE);
        setAlignmentX(CENTER_ALIGNMENT);
      }
    }

    private class CustomButton extends JButton {

      CustomButton(String label, Font font) {
        super(label);
        setFocusable(false);
        setFont(font);
        setAlignmentX(CENTER_ALIGNMENT);
        setBackground(new Color(40, 40, 70));
        setForeground(Color.WHITE);
      }
    }

    private final Label[] labels = new Label[results.length];

    StatsPane() {
      setLayout(new BoxLayout(this, BoxLayout.Y_AXIS));
      final Dimension size = new Dimension(SIDE_LENGTH, SIDE_LENGTH);
      setPreferredSize(size);
      setMaximumSize(size);
      setMinimumSize(size);
      setBackground(COLOUR);

      add(Box.createVerticalGlue());

      final Font font = new Font("Barlow", Font.PLAIN, 20);
      for (int i = 0; i < labels.length; i++) {
        labels[i] = new Label(i + ": N/A", font);
        add(labels[i]);
      }

      final JButton resetButton = new CustomButton("Reset", font);
      resetButton.addActionListener(e -> {
        for (double[] row : image) Arrays.fill(row, 0);
        draw.repaint();
      });
      add(resetButton);

      final JButton newTrainerButton = new CustomButton("New Trainer", font);
      newTrainerButton.addActionListener(e -> newTrainer());
      add(newTrainerButton);

      final JButton loadTrainerButton = new CustomButton("Load Trainer", font);
      loadTrainerButton.addActionListener(e -> loadTrainer());
      add(loadTrainerButton);

      add(Box.createVerticalGlue());
    }

    public void update() {
      record IndexedValue(double value, int index) {}
      final IndexedValue[] values = new IndexedValue[results.length];
      for (int i = 0; i < results.length; i++) values[i] =
        new IndexedValue(results[i], i);
      Arrays.sort(values, (v1, v2) -> Double.compare(v2.value, v1.value));

      for (int i = 0; i < values.length; i++) {
        labels[i].setText(
            String.format("%d: %.2f%%", values[i].index, values[i].value * 100)
          );
      }
    }

    private void newTrainer() {
      try {
        mouseAdapter.trainer =
          new App<Byte>()
            .start(new int[] { App.IMAGE_SIZE, 256, 64, 16, 10 }, 50_000);
        final JFileChooser chooser = new JFileChooser();
        if (chooser.showSaveDialog(this) == JFileChooser.APPROVE_OPTION) {
          try (
            ObjectOutputStream oos = new ObjectOutputStream(
              new FileOutputStream(chooser.getSelectedFile())
            )
          ) {
            oos.writeObject(mouseAdapter.trainer);
          }
        }
      } catch (IOException ex) {
        ex.printStackTrace();
      }
    }

    @SuppressWarnings("unchecked")
    private void loadTrainer() {
      final JFileChooser chooser = new JFileChooser();
      if (
        chooser.showOpenDialog(DisplayFrame.this) == JFileChooser.APPROVE_OPTION
      ) {
        try (
          ObjectInputStream oos = new ObjectInputStream(
            new FileInputStream(chooser.getSelectedFile())
          )
        ) {
          mouseAdapter.trainer = (NetworkTrainer<Byte>) oos.readObject();
        } catch (IOException | ClassNotFoundException ex) {
          ex.printStackTrace();
        }
      }
    }
  }

  public DisplayFrame() {
    setDefaultCloseOperation(EXIT_ON_CLOSE);
    setLayout(new BoxLayout(getContentPane(), BoxLayout.X_AXIS));
    setSize(SIDE_LENGTH * 2 + 100, 800);
    getContentPane().setBackground(COLOUR);
    setLocationRelativeTo(null);
    getContentPane().add(Box.createHorizontalGlue());
    mouseAdapter = new CustomMouseAdapter();
    draw = new DrawPane();
    getContentPane().add(draw);
    stats = new StatsPane();
    getContentPane().add(stats);
    getContentPane().add(Box.createHorizontalGlue());
    setVisible(true);
  }
}
